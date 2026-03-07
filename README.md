cos7 线粒体超分辨图像 Resunet 模型
1.tran8bit 用cv2将16bit图像转为8bit
2. data_load 载入图像原数据及对应标签数据
3. train 训练模型

train_cv_cbamselect.py --cbam_pos  (每张图减自身均值)

###### 推理脚本
python predict_tif_and_timeseries_cbam_perimgmean.py \
  --test_dir /home/CWB/test_tif \
  --out_dir /home/CWB/pred_result
#######
