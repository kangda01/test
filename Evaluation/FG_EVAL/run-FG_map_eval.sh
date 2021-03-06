python map_eval_FG_cv2.py \
        --annotation_file ./results/GTBOX_GIS-SIGN-FG-BASE2+.txt \
        --detection_file  ./results/DECTS_TCV187-SSCE-TOP3-C1-90_GIS-SIGN-FG-BASE2+.txt \
        --confidence      0.6 \
        --iou             0.4 \
        --input_dir       /data/TESTSET/GIS-SIGN-FG_baseline/WHOLE_SET/images \
        --out_dir         ./results \
        --record_mistake  False \
        --draw_full_img   False \
        --draw_cut_box    False \
        --detect_subclass True
