import requests
import logging

logger = logging.getLogger(__name__)

def image_upload(file_path):
    url = "https://web.aizex.top/api/upload-oss"

    params = {
        "fileType": "file"
    }

    headers = {
        'Accept': 'application/json',
        'Accept-Language': 'zh-CN,zh;q=0.9,zh-TW;q=0.8',
        'Authorization': 'Bearer sk-hsqdnUf9obbq3WjM5lo2ixoRX61q1mDPhuoV9aNgcpVvIRq9',
        'model': 'mj-chat',
        'Origin': 'https://web.aizex.top',
        'Referer': 'https://web.aizex.top/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'
    }

    try:
        files = {
            'file': ('image.jpg', open(file_path, 'rb'), 'image/jpeg')
        }

        response = requests.post(url, params=params, headers=headers, files=files)

        response.raise_for_status()

        logger.info("successful request: %s", response.status_code)
        try:
            logger.info("response content: %s", response.json())
            url = response.json()['fileUrl']
        except requests.exceptions.JSONDecodeError:
            logger.info("response content %s:", response.text)
            url = eval(response.text)['fileUrl']

    except FileNotFoundError:
        logger.info(f"error: can not find file - {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"failed request: {e}")
    finally:
        if 'files' in locals() and files['file'][1]:
            files['file'][1].close()

    return url
