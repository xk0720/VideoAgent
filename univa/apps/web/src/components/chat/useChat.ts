"use client";

import { useState, useRef, useEffect, useCallback } from 'react';
import { Message, ChatState, GeneratedFile, MediaReference, MediaSelectorState, TodoItem } from './types';
import { MediaCacheService } from './utils/mediaCacheService';
import { parseMediaReferences, addMediaReferencesToText } from './utils/mediaReferenceParser';
import { useProjectStore } from '@/stores/project-store';

const CHAT_STORAGE_KEY_PREFIX = 'ai-chat-state-';
const ACCESS_CODE_STORAGE_KEY = 'ai-chat-access-code';

// Access code status interface
export interface AccessCodeStatus {
  enabled: boolean;
  usage_count: number;
  conversation_count: number;
  max_conversations: number | null;
  remaining_conversations: number | null;
  last_used: string | null;
  created_at: string | null;
}

export const useChat = () => {
  // Get current project ID
  const activeProject = useProjectStore((state) => state.activeProject);
  const projectId = activeProject?.id;
  // Media cache service instance
  const mediaCacheService = useRef(new MediaCacheService()).current;

  // Restore access code from localStorage
  const getStoredAccessCode = (): string => {
    if (typeof window !== 'undefined') {
      try {
        return localStorage.getItem(ACCESS_CODE_STORAGE_KEY) || '';
      } catch (error) {
        console.warn('Failed to restore access code:', error);
      }
    }
    return '';
  };

  // Restore state from localStorage (based on project ID)
  const getInitialState = (currentProjectId?: string): ChatState => {
    if (typeof window !== 'undefined' && currentProjectId) {
      try {
        const storageKey = `${CHAT_STORAGE_KEY_PREFIX}${currentProjectId}`;
        const saved = localStorage.getItem(storageKey);
        if (saved) {
          const parsed = JSON.parse(saved);
          return {
            ...parsed,
            isLoading: false, // Reset loading state
            error: null, // Reset error state
            mediaSelector: parsed.mediaSelector || {
              isOpen: false,
              position: { x: 0, y: 0 },
              searchQuery: '',
              selectedIndex: 0,
            },
            referencedMedia: parsed.referencedMedia || [],
            accessCode: getStoredAccessCode(), // Restore access code
          };
        }
      } catch (error) {
        console.warn('Failed to restore chat state:', error);
      }
    }
    return {
      messages: [],
      inputText: '',
      isLoading: false,
      sessionId: null,
      error: null,
      mediaSelector: {
        isOpen: false,
        position: { x: 0, y: 0 },
        searchQuery: '',
        selectedIndex: 0,
      },
      referencedMedia: [],
      connectionStatus: 'disconnected',
      retryCount: 0,
      maxRetries: 3,
      accessCode: getStoredAccessCode(),
    };
  };

  const [state, setState] = useState<ChatState>(() => getInitialState(projectId));

  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Save state to localStorage (based on project ID)
  const saveState = (newState: ChatState, currentProjectId?: string) => {
    if (typeof window !== 'undefined' && currentProjectId) {
      try {
        const stateToSave = {
          ...newState,
          isLoading: false, // Don't save loading state
          error: null, // Don't save error state
        };
        const storageKey = `${CHAT_STORAGE_KEY_PREFIX}${currentProjectId}`;
        localStorage.setItem(storageKey, JSON.stringify(stateToSave));
      } catch (error) {
        console.warn('Failed to save chat state:', error);
      }
    }
  };

  // Listen for state changes and save to current project
  useEffect(() => {
    if (projectId) {
      saveState(state, projectId);
    }
  }, [state.messages, state.inputText, state.sessionId]);

  // Listen for project ID changes and load chat history from localStorage
  useEffect(() => {
    if (projectId) {
      console.log('Loading chat state for project:', projectId);
      const newState = getInitialState(projectId);
      setState(newState);
    }
  }, [projectId]);


  // Stream response with reconnection mechanism
  const streamResponse = (prompt: string, displayText?: string) => {
    const maxRetries = 3;
    let retryCount = 0;
    let isCompleted = false;
    
    // Create task execution flow state
    let currentTodoItems: TodoItem[] = [];
    let currentOverallDescription = '';
    
    // Counter for generating unique IDs
    let messageIdCounter = 0;
    const generateUniqueId = () => {
      messageIdCounter++;
      return `msg-${Date.now()}-${messageIdCounter}-${Math.floor(Math.random() * 1000000)}`;
    };

    const connectSSE = () => {
      try {
        setState(prev => ({
          ...prev,
          isLoading: true,
          error: null,
          connectionStatus: retryCount === 0 ? 'connecting' : 'reconnecting',
          retryCount,
          maxRetries,
        }));

        // Use the prompt with already replaced local paths, no need to re-parse and rebuild
        const fullPrompt = prompt;

        // Only add user message on first connection
        if (retryCount === 0) {
          const userMessage: Message = {
            id: `user-${Date.now()}-${Math.floor(Math.random() * 1000000)}`,
            role: 'user',
            content: displayText || prompt,
            timestamp: new Date().toISOString(),
          };

          setState(prev => ({
            ...prev,
            messages: [...prev.messages, userMessage],
            inputText: '',
            referencedMedia: [],
          }));
        }

        // Establish SSE connection with full prompt containing media references
        // Note: EventSource doesn't support custom headers, so we pass access code as URL parameter
        let url = `/api/chat/stream?prompt=${encodeURIComponent(fullPrompt)}${state.sessionId ? `&sessionId=${state.sessionId}` : ''}`;
        if (state.accessCode) {
          url += `&accessCode=${encodeURIComponent(state.accessCode)}`;
        }
        eventSourceRef.current = new EventSource(url);
        
        // Set heartbeat timeout detection - if no message received within 90 seconds, attempt reconnection
        if (heartbeatTimeoutRef.current) {
          clearTimeout(heartbeatTimeoutRef.current);
        }
        heartbeatTimeoutRef.current = setTimeout(() => {
          console.warn('No heartbeat received for 90 seconds, attempting reconnection...');
          if (!isCompleted && retryCount < maxRetries) {
            attemptReconnect();
          }
        }, 90000);

        eventSourceRef.current.onmessage = (event) => {
          try {
            // Reset heartbeat timeout
            if (heartbeatTimeoutRef.current) {
              clearTimeout(heartbeatTimeoutRef.current);
              heartbeatTimeoutRef.current = setTimeout(() => {
                console.warn('No heartbeat received for 90 seconds, attempting reconnection...');
                if (!isCompleted && retryCount < maxRetries) {
                  attemptReconnect();
                }
              }, 90000);
            }

            const data = JSON.parse(event.data);
            
            // Handle heartbeat message
            if (data.type === 'heartbeat') {
              console.log('Heartbeat received');
              // Update connection status to connected
              setState(prev => ({
                ...prev,
                connectionStatus: 'connected',
                error: null,
              }));
              return;
            }
            
            if (data.type === 'content') {
                const progressMessage: Message = {
                  id: generateUniqueId(),
                  role: 'assistant',
                  content: data.content,
                  timestamp: new Date().toISOString(),
                  messageType: 'assistant',
                };
                
                setState(prev => ({
                  ...prev,
                  messages: [...prev.messages, progressMessage],
                }));
            } else if (data.type === 'tool_start') {
              const toolStartMessage: Message = {
                id: generateUniqueId(),
                role: 'assistant',
                content: `Executing tool: ${data.tool}. Please don't close this window.`,
                timestamp: new Date().toISOString(),
                messageType: 'tool_start',
              };
              
              setState(prev => ({
                ...prev,
                messages: [...prev.messages, toolStartMessage],
              }));
            } else if (data.type === 'tool_end') {
              const toolResult = data.result;
              let resultSummary = 'Tool execution completed';
              let toolEndContent = '';
              let generatedFiles: GeneratedFile[] = [];
              
              if (toolResult && typeof toolResult === 'object') {
                if (toolResult.success === 'True') {
                  resultSummary = 'Tool execution successful';
                } else {
                  resultSummary = 'Tool execution failed';
                }
                
                toolEndContent = resultSummary;
                
                if (toolResult.content && toolResult.content.trim()) {
                  toolEndContent += `\n\n${toolResult.content}`;
                }
                
                // Get file path from output_path, supports single file or file array
                if (toolResult.output_path && toolResult.output_path.trim()) {
                  const outputPaths = Array.isArray(toolResult.output_path)
                    ? toolResult.output_path
                    : [toolResult.output_path];
                  
                  // Filter out empty strings
                  const validPaths = outputPaths.filter((path: string) => path && path.trim());
                  
                  validPaths.forEach((filePath: string) => {
                    const fileName = filePath.split('/').pop() || filePath;
                    const fileExtension = fileName.split('.').pop()?.toLowerCase() || '';
                    
                    let fileType: 'video' | 'image' | 'audio' = 'video';
                    if (['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'].includes(fileExtension)) {
                      fileType = 'image';
                    } else if (['wav', 'mp3', 'aac', 'ogg', 'flac'].includes(fileExtension)) {
                      fileType = 'audio';
                    }
                    
                    generatedFiles.push({
                      path: filePath,
                      type: fileType,
                      name: filePath
                    });
                  });
                  
                  if (validPaths.length === 1) {
                    toolEndContent += `\n\nOutput Files: ${validPaths[0]}`;
                  } else if (validPaths.length > 1) {
                    toolEndContent += `\n\nTotal (${validPaths.length}): ${validPaths.join(', ')}`;
                  }
                }
              } else if (typeof toolResult === 'string') {
                try {
                  const parsedResult = JSON.parse(toolResult);
                  if (parsedResult.success) {
                    resultSummary = 'Tool execution successful';
                  } else {
                    resultSummary = 'Tool execution failed';
                  }
                  
                  // Get file path from output_path, supports single file or file array
                  if (parsedResult.output_path) {
                    const outputPaths = Array.isArray(parsedResult.output_path) 
                      ? parsedResult.output_path 
                      : [parsedResult.output_path];
                    
                    outputPaths.forEach((filePath: string) => {
                      const fileName = filePath.split('/').pop() || filePath;
                      const fileExtension = fileName.split('.').pop()?.toLowerCase() || '';
                      
                      let fileType: 'video' | 'image' | 'audio' = 'video';
                      if (['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'].includes(fileExtension)) {
                        fileType = 'image';
                      } else if (['wav', 'mp3', 'aac', 'ogg', 'flac'].includes(fileExtension)) {
                        fileType = 'audio';
                      }
                      
                      generatedFiles.push({
                        path: filePath,
                        type: fileType,
                        name: filePath  // Use full path as display name
                      });
                    });
                  }
                  
                  toolEndContent = JSON.stringify(parsedResult, null, 2);
                } catch (e) {
                  toolEndContent = `Tool execution completed: ${toolResult}`;
                }
              }
              
              const toolEndMessage: Message = {
                id: generateUniqueId(),
                role: 'assistant',
                content: toolEndContent,
                timestamp: new Date().toISOString(),
                messageType: 'tool_end',
                generatedFiles: generatedFiles.length > 0 ? generatedFiles : undefined,
              };
              
              setState(prev => ({
                ...prev,
                messages: [...prev.messages, toolEndMessage],
              }));
            } else if (data.type === 'todo_progress') {
              currentTodoItems = data.items || [];
              currentOverallDescription = data.overall_description || '';
              
              const todoProgressMessage: Message = {
                id: generateUniqueId(),
                role: 'assistant',
                content: 'Task progress updated',
                timestamp: new Date().toISOString(),
                messageType: 'todo_progress',
                todoItems: currentTodoItems,
                overallDescription: currentOverallDescription,
              };
              
              setState(prev => ({
                ...prev,
                messages: [...prev.messages, todoProgressMessage],
              }));
            } else if (data.type === 'finish') {
              isCompleted = true;
              
              const completionMessage: Message = {
                id: generateUniqueId(),
                role: 'assistant',
                content: 'Task execution completed!',
                timestamp: new Date().toISOString(),
                messageType: 'completion',
                todoItems: currentTodoItems.map(item => ({
                  ...item,
                  status: 'completed'
                })),
                overallDescription: currentOverallDescription,
              };
              
              setState(prev => ({
                ...prev,
                messages: [...prev.messages, completionMessage],
                isLoading: false,
                sessionId: data.session_id || prev.sessionId,
                connectionStatus: 'disconnected',
              }));
              
              cleanupConnection();
            } else if (data.type === 'error') {
              throw new Error(data.content);
            }
          } catch (error) {
            console.error('Error processing stream data:', error);
            setState(prev => ({
              ...prev,
              error: error instanceof Error ? error.message : 'Error processing response',
              isLoading: false,
            }));
            
            cleanupConnection();
          }
        };

        eventSourceRef.current.onerror = async (error) => {
          console.error('SSE error:', error);
          
          // Try to get response status from EventSource
          // EventSource triggers error event when encountering HTTP errors
          // We need to re-request to get specific error information
          try {
            const testResponse = await fetch(url.replace('/api/chat/stream', '/api/chat/stream'));
            if (!testResponse.ok) {
              let errorMessage = 'Connection error';
              
              if (testResponse.status === 401) {
                errorMessage = '❌ Access code not provided. Please configure your access code in the project settings.';
              } else if (testResponse.status === 403) {
                errorMessage = '❌ Invalid access code. The access code you provided is incorrect or has been disabled. Please check your access code in the project settings.';
              }
              
              setState(prev => ({
                ...prev,
                error: errorMessage,
                isLoading: false,
                connectionStatus: 'disconnected',
              }));
              
              cleanupConnection();
              return;
            }
          } catch (fetchError) {
            console.error('Error checking response status:', fetchError);
          }
          
          // Check if it's a timeout error
          const isTimeoutError = error && (error as any).code === 23; // TIMEOUT_ERR
          
          if (isTimeoutError && !isCompleted && retryCount < maxRetries) {
            console.log('Detected timeout error, attempting reconnection...');
            attemptReconnect();
          } else {
            // For other errors or when max retries reached, display error message
            setState(prev => ({
              ...prev,
              error: isCompleted
                ? 'Task completed but connection was lost.'
                : retryCount >= maxRetries
                  ? 'Connection failed after multiple retries. The task may still be running in the background. Please refresh the page to check results.'
                  : 'Connection lost. If the task was running, it may still be processing in the background.',
              isLoading: false,
            }));
            
            cleanupConnection();
          }
        };

      } catch (error) {
        console.error('Error starting stream:', error);
        setState(prev => ({
          ...prev,
          error: error instanceof Error ? error.message : 'Failed to start streaming',
          isLoading: false,
        }));
      }
    };

    const attemptReconnect = () => {
      if (isCompleted || retryCount >= maxRetries) {
        setState(prev => ({
          ...prev,
          error: retryCount >= maxRetries ? 'Connection failed after multiple retries. The task may still be running in the background. Please refresh to check results.' : 'Connection error',
          isLoading: false,
          connectionStatus: 'disconnected',
        }));
        cleanupConnection();
        return;
      }

      retryCount++;
      console.log(`Attempting to reconnect (${retryCount}/${maxRetries})...`);
      
      setState(prev => ({
          ...prev,
          error: `Connection lost. Reconnecting... (${retryCount}/${maxRetries}). Task continues running in background.`,
          connectionStatus: 'reconnecting',
          retryCount,
          maxRetries,
        }));

      cleanupConnection();
      
      // Wait a while before reconnecting, using exponential backoff
      const delay = Math.min(2000 * Math.pow(1.5, retryCount - 1), 10000); // 2s, 3s, 4.5s, 6.75s, 10s
      reconnectTimeoutRef.current = setTimeout(() => {
        // Use the same sessionId when reconnecting to restore task state
        connectSSE();
      }, delay);
    };

    const cleanupConnection = () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      
      if (heartbeatTimeoutRef.current) {
        clearTimeout(heartbeatTimeoutRef.current);
        heartbeatTimeoutRef.current = null;
      }
      
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };

    // Start connection
    connectSSE();
  };

  // Clean up resources
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (heartbeatTimeoutRef.current) {
        clearTimeout(heartbeatTimeoutRef.current);
      }
    };
  }, []);

  // Handle input change
  const handleInputChange = (value: string) => {
    setState(prev => ({
      ...prev,
      inputText: value,
    }));
  };

  // Handle send message
  const handleSend = async () => {
    if (state.inputText.trim() && !state.isLoading) {
      // Check if access code is set
      if (!state.accessCode || state.accessCode.trim() === '') {
        setState(prev => ({
          ...prev,
          error: 'Please configure the access code in the project settings before sending messages.',
        }));
        return;
      }

      const originalText = state.inputText.trim(); // Save original input for display
      let messageText = originalText; // Text to send to backend
      
      // If there are referenced media files, cache them locally first
      if (state.referencedMedia.length > 0) {
        try {
          setState(prev => ({
            ...prev,
            isLoading: true,
            error: null,
          }));

          // Cache all referenced media files
          const cacheResults = await mediaCacheService.cacheMediaFiles(state.referencedMedia);
          
          // Replace media reference paths in message with local cache paths (only for sending to backend)
          for (const [mediaId, localPath] of cacheResults) {
            const mediaRef = state.referencedMedia.find(m => m.id === mediaId);
            if (mediaRef) {
              const oldReference = `@[${mediaRef.name}](${mediaId})`;
              const newReference = `@[${mediaRef.name}](${localPath})`;
              messageText = messageText.replace(oldReference, newReference);
            }
          }

          setState(prev => ({
            ...prev,
            isLoading: false,
          }));
        } catch (error) {
          console.error('Failed to cache media files:', error);
          setState(prev => ({
            ...prev,
            error: 'Failed to cache media files',
            isLoading: false,
          }));
          return;
        }
      }
      
      streamResponse(messageText, originalText);
    }
  };

  // Handle keyboard event
  const handleKeyDown = async (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      await handleSend();
    }
  };

  // Clear chat history
  const clearChat = () => {
    const clearedState = {
      messages: [],
      inputText: '',
      isLoading: false,
      sessionId: null,
      error: null,
      mediaSelector: {
        isOpen: false,
        position: { x: 0, y: 0 },
        searchQuery: '',
        selectedIndex: 0,
      },
      referencedMedia: [],
      connectionStatus: 'disconnected' as const,
      retryCount: 0,
      maxRetries: 3,
      accessCode: state.accessCode, // Keep access code
    };
    setState(clearedState);
    if (typeof window !== 'undefined' && projectId) {
      const storageKey = `${CHAT_STORAGE_KEY_PREFIX}${projectId}`;
      localStorage.removeItem(storageKey);
    }
  };

  // Handle media reference
  const handleMediaReference = (media: MediaReference) => {
    setState(prev => ({
      ...prev,
      referencedMedia: [...prev.referencedMedia.filter(m => m.id !== media.id), media],
    }));
  };

  // Remove media reference
  const removeMediaReference = (mediaId: string) => {
    setState(prev => ({
      ...prev,
      referencedMedia: prev.referencedMedia.filter(m => m.id !== mediaId),
      inputText: prev.inputText.replace(new RegExp(`@\[[^\]]*\]\(${mediaId}\)\s?`, 'g'), ''),
    }));
  };

  // Handle access code change
  const handleAccessCodeChange = (code: string) => {
    setState(prev => ({
      ...prev,
      accessCode: code,
    }));
    
    // Save to localStorage
    if (typeof window !== 'undefined') {
      try {
        if (code) {
          localStorage.setItem(ACCESS_CODE_STORAGE_KEY, code);
        } else {
          localStorage.removeItem(ACCESS_CODE_STORAGE_KEY);
        }
      } catch (error) {
        console.warn('Failed to save access code:', error);
      }
    }
  };

  // Get access code status - use useCallback to avoid infinite loops
  const fetchAccessCodeStatus = useCallback(async (): Promise<AccessCodeStatus | null> => {
    if (!state.accessCode) {
      return null;
    }

    try {
      const response = await fetch('/api/access-code/status', {
        method: 'GET',
        headers: {
          'X-Access-Code': state.accessCode,
        },
      });

      if (!response.ok) {
        if (response.status === 401 || response.status === 404) {
          console.warn('Access code is invalid or not found');
          return null;
        }
        throw new Error(`Failed to fetch access code status: ${response.statusText}`);
      }

      const status: AccessCodeStatus = await response.json();
      return status;
    } catch (error) {
      console.error('Error fetching access code status:', error);
      return null;
    }
  }, [state.accessCode]);

  return {
    messages: state.messages,
    inputText: state.inputText,
    isLoading: state.isLoading,
    error: state.error,
    sessionId: state.sessionId,
    referencedMedia: state.referencedMedia,
    connectionStatus: state.connectionStatus,
    retryCount: state.retryCount,
    maxRetries: state.maxRetries,
    accessCode: state.accessCode || '',
    handleInputChange,
    handleSend,
    handleKeyDown,
    clearChat,
    handleMediaReference,
    removeMediaReference,
    handleAccessCodeChange,
    fetchAccessCodeStatus,
  };
};