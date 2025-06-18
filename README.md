## 中山大學114級資工系 畢業專題製作
### 題目: AI生成語音辨識及應用
#### 成員: 3位資工系同學
### 簡介
使用**KV260當作嵌入式系統**來實現即時收音與辨識是否有AI生成語音(中英文)的存在，進而防止詐騙嫌犯使用AI生成語音混淆受害者。  
其中**模型的量化與編譯**，然後**移植模型**到KV260上運行是藉由[Vitis-AI](https://docs.amd.com/r/en-US/ug1414-vitis-ai/Vitis-AI-Overview)這個工具進行。  
警告通知使用LINE來實現，使用Flask+HTML去設計使用者介面，可以控制錄音與結果顯示，線上應用中我的部分是**設計網頁來運用模型的功能。**
### 架構圖
![image](project_structure.png)

### 網頁線上應用
[Repo](https://github.com/AnnTaiwan/flask-ngrok-ml)  
可以上傳音檔和影片或是URL去辨識是否為AI生成語音，另外使用Gemini LLM去對上傳內容判斷有無詐騙風險。
### 詳細作法筆記 on HackMD
1. [kv260板子上的各種操作](https://hackmd.io/@NsysuAnn/H1F9KKYU0)
2. [Vitis_AI](https://hackmd.io/@NsysuAnn/Hk1J81EGC)
3. 網頁頁面實作: [利用Flask部屬模型到web上](https://hackmd.io/@NsysuAnn/SkjrN2e9C)
4. [中文語音模型訓練實驗記錄](https://hackmd.io/@NsysuAnn/rkkSEvIL0)
4. [英文語音模型訓練實驗記錄](https://hackmd.io/@NsysuAnn/ryLvAgCF0)

### Demo video
[![Watch the video](https://img.youtube.com/vi/tRtrg5NVRNE/0.jpg)](https://www.youtube.com/watch?v=tRtrg5NVRNE)