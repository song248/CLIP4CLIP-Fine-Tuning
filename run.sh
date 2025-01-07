python preprocess/compress_video.py --input_root [raw_video_path] --output_root [compressed_video_path]


--features_path is the video root path

--linear_patch can be set with 2d or 3d

--sim_header can be set with meanP, seqLSTM, seqTransf, or tightTransf

--pretrained_clip_name can be set with ViT-B/32 or ViT-B/16

--resume_model can be used to reload the saved optimizer state to continuely train the model, Note: need to set the corresponding chechpoint via --init_model simultaneously.


DATA_PATH='/home/song/Desktop/CLIP4Clip/data'
python -m torch.distributed.launch --nproc_per_node=2 \
main_task_retrieval.py --do_train --num_thread_reader=2 \
--epochs=5 --batch_size=16 --n_display=50 \
--data_path 'data' \
--features_path 'data/videos' \
--output_dir ckpts/ckpt_msvd_retrieval_looseType \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msvd \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32


DATA_PATH='/home/song/Desktop/CLIP4Clip/data'
python -m torch.distributed.launch --nproc_per_node=2 \
main_task_retrieval.py --do_train --num_thread_reader=2 \
--epochs=10 --batch_size=64 --n_display=50 \
--data_path 'data' \
--features_path 'data/videos' \
--output_dir ckpts/ckpt_msvd_retrieval_looseType \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msvd \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32