## 即時錄音在mobaXterm環境下操作KV260
### 整體流程
1. 先打開兩個terminal
2. 在各自的terminal下打
* `sh recording_gstreamer.sh`: 開始錄音，會自動五秒就切一段，其中程式檔中是寫6秒，結果錄製的音檔會是五秒，但是錄起來兩段音檔中間會少差不多不到一秒(感覺是無可避免的)
*  `sh ENG_predict_audio_real_time.sh` or `sh_CH_predict_audio_real_time.sh`會一直等待`audio`資料夾中有無新的音檔，有就去做後續的計算與辨識

### 詳細解釋
#### `ENG_predict_audio_real_time.sh`中的流程:
1. 有一個迴圈，隨時查看`audio` folder中有無音檔，有的話就將它先行降噪  (`sh denoise_add_volume_ENG.sh`)
2. 將降噪完的音檔利用`cal_mel_spec_ver4`去計算頻譜值，其中我原先的設計**為當第一段不管有無5秒皆會去計算並做補值的操作**，因此套用到現在的音檔一定不超過五秒，也是可以運作，因此就沒更改了
3. 然後使用`plot_mel_spec_from_txt_scale_ver2`去繪製圖片，英文的會繪製成[496 x 371]，則中文的是使用`plot_mel_spec_from_txt_ver2`會繪製成[216 x 128]
4. 執行`./demo_final_segment_audio_ver2 CNN_model9_CH_ver4_netname.xmodel $IMG_FOLDER/ detail_result_CH.txt`
當圖片都繪製好後即可去預測，現在做的流程只會一次一張圖片去做預測，因此只會看到預測一次的結果
5. 然後每次預測完，都將結果儲存成`result.txt`，原先是要將此音檔的結果透過傳LINE傳給使用者，想說這個功能照舊，等於一個音檔可能會傳遞多次通知，我覺得這樣很符合實際應用，即時傳通知的概念
6. 當傳完通知後，**會先將當前處理的音檔移到另一個資料夾(`processed_audio`)中，以免重複預測，即可進入迴圈的前頭**，檢查有無新的音檔出現，沒有的話，就等待`sleep 5`，再檢查一次。**如果有的話就回到第1步**。