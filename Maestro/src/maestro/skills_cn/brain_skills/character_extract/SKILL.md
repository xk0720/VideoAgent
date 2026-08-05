---
name: character_extract
agent: window brain(pipeline/window_loop.py 中的 §A1——在分镜之前、针对剧本运行)
description: 把全部角色提取为规范的 "static:/dynamic:" 外观契约。给定角色的外观逐字取自图片描述。严格 JSON 输出。
---

# 角色提取——建立角色阵容正典

## 职责
每个相关角色都要生成一条正典条目
`"static: <look>; dynamic: <what varies>"`。在下游,REVIEWER(审阅者)
就是拿这份正典来评判角色身份的一致性,肖像生成器也从这份正典取材作画。
对于拥有参考图的角色,这份正典绝不会进入视频 prompt——在那里,
身份由图片的像素本身来承载。

## 规则

1. ONE ENTRY PER ENTITY(每个实体一条):把各种别名合并为同一个角色
   (选最有用的那个名字)。背景群演和人群不算角色。
2. UNNAMED CHARACTERS(无名角色):使用一个稳定的职业/特征别名
   ("the barista"),可在镜头描述中作为标记反复使用。
3. STATIC vs DYNAMIC(静态与动态):`static:` = 近乎不变的身份特征
   (体格、面容、发型、肤色、标志性服装——必须带颜色);`dynamic:` =
   会变化的部分(姿势、表情、手持道具)。
4. GIVEN-CHARACTERS LAW(给定角色法则):任务 JSON 可能带有
   `given_characters`——即用户已绑定到官方图片的角色名,每个名字都
   附带一条 `image_look` 描述,由视觉模型根据真实图片写成。这些名字是
   权威键名:必须逐字采用每一个(绝不改名、绝不翻译、绝不丢弃)。
   图片是其 `static:` 那一半的唯一来源——`image_look` 里的颜色和
   服装用词必须一字不差地照抄。绝不添加 `image_look` 没有写明的外观
   细节(如果它没说外套的颜色,就写 "military coat",而不是
   "white military coat")。剧本与 `image_look` 有冲突时,
   以 `image_look` 为准。剧本只负责提供 `dynamic:` 部分。
   (之后有一道确定性闸门会把图片描述重新强制写回——所以第一次
   就要写对。)明显指向某个给定角色的剧本别名,要合并进该给定名字;
   两个剧本角色也可能共用同一张给定图片——此时两个名字都要输出。
5. FILL GAPS PLAUSIBLY — SCRIPT-ONLY CHARACTERS ONLY(合理补全空缺——
   仅限纯剧本角色):对没有图片的角色,要发明具体、连贯的外观
   (明确的颜色、具体的特征)——单薄的描述锚不住身份。这条规则
   绝不适用于给定角色:一个与其图片相矛盾的臆造细节,会毒害下游的
   每一次审查。
6. DISTINCTNESS(区分度):把纯剧本角色的外观彼此拉开距离,也要与
   给定角色阵容拉开距离(不同的发型、服装颜色、体格)。
7. VISUAL WORDS ONLY(只用视觉词汇):描述词一律用英文(名字可以
   保留用户的语言)。不得使用真实名人或 IP。

## 输出格式(严格 JSON——只输出它,不输出任何其他内容)

{"characters": {"<name>": "static: <traits>; dynamic: <traits>"}}
