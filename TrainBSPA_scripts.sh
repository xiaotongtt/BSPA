## train
# BI, scale 2, 3, 4, 8
##################################################################################################################################
# BI, scale 2, 3, 4, 8
# RCAN_BIX2_G10R20P48, input=48x48, output=96x96
#LOG=./../experiment/RCAN_BIX2_G10R20P48-`date +%Y-%m-%d-%H-%M-%S`.txt
#CUDA_VISIBLE_DEVICES=0 python main.py --model RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 96 2>&1 | tee $LOG
#
## RCAN_BIX3_G10R20P48, input=48x48, output=144x144
#LOG=./../experiment/RCAN_BIX3_G10R20P48-`date +%Y-%m-%d-%H-%M-%S`.txt
#CUDA_VISIBLE_DEVICES=0 python main.py --model RCAN --save RCAN_BIX3_G10R20P48 --scale 3 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt 2>&1 | tee $LOG
#
## RCAN_BIX4_G10R20P48, input=48x48, output=192x192
#LOG=./../experiment/RCAN_BIX4_G10R20P48-`date +%Y-%m-%d-%H-%M-%S`.txt
#CUDA_VISIBLE_DEVICES=0 python main.py --model RCAN --save RCAN_BIX4_G10R20P48 --scale 4 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pt 2>&1 | tee $LOG
#
## RCAN_BIX8_G10R20P48, input=48x48, output=384x384
#LOG=./../experiment/RCAN_BIX8_G10R20P48-`date +%Y-%m-%d-%H-%M-%S`.txt
#CUDA_VISIBLE_DEVICES=0 python main.py --model RCAN --save RCAN_BIX8_G10R20P48 --scale 8 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt 2>&1 | tee $LOG


# RCAN
#CUDA_VISIBLE_DEVICES=0 python main_filter_edsr.py --model edsr_baseline --scale 4 --batch_size 16 --un_factor '10' --lr 2e-4 --decay_type 'step' --save ../experiment/EDSR/EDSR_x4_WA_IG_filter_un10_0.01_Sy8  --remain_factor 0.01 # --test_only #--repeat 20 --ext sep
CUDA_VISIBLE_DEVICES=0 python main_weight_edsr.py --model rcan --scale 2 --batch_size 16 --patch_size 128 --print_every 10 --un_factor '10' --lr 2e-4 --decay_type 'step' --save ../experiment/RCAN/RCAN_x2_WA_IG_filter_un10_Sy10 --remain_factor 0.1  --record_filters_folder ../experiment/RCAN/RCAN_x2_WA_IG_filter_un10_Sy10
CUDA_VISIBLE_DEVICES=0 python main_weight_edsr.py --model rcan --scale 3 --batch_size 16 --patch_size 129 --print_every 10 --un_factor '10' --lr 2e-4 --decay_type 'step' --save ../experiment/RCAN/RCAN_x3_WA_IG_filter_un10_Sy10 --remain_factor 0.1  --record_filters_folder ../experiment/RCAN/RCAN_x3_WA_IG_filter_un10_Sy10
CUDA_VISIBLE_DEVICES=0 python main_weight_edsr.py --model rcan --scale 4 --batch_size 16 --patch_size 128 --print_every 10 --un_factor '10' --lr 2e-4 --decay_type 'step' --save ../experiment/RCAN/RCAN_x4_WA_IG_filter_un10_Sy10 --remain_factor 0.1  --record_filters_folder ../experiment/RCAN/RCAN_x4_WA_IG_filter_un10_Sy10

# FSRCNN
CUDA_VISIBLE_DEVICES=1 python main_weight_fsrcnn.py --model fsrcnn --scale 4 --batch_size 16 --patch_size 128 --print_every 10 --un_factor '10' --lr 2e-4 --decay_type 'step' --save ../experiment/FSRCNN/FSRCNN_x4_IG_filter_un10_RF0.1_Sy10 --remain_factor 0.1  --record_filters_folder ../experiment/FSRCNN/FSRCNN_x4_IG_un10_RF0.1_Sy10/IG/
CUDA_VISIBLE_DEVICES=1 python main_weight_fsrcnn.py --model fsrcnn --scale 3 --batch_size 16 --patch_size 129 --print_every 10 --un_factor '10' --lr 2e-4 --decay_type 'step' --save ../experiment/FSRCNN/FSRCNN_x3_IG_filter_un10_RF0.1_Sy10 --remain_factor 0.1  --record_filters_folder ../experiment/FSRCNN/FSRCNN_x3_IG_un10_RF0.1_Sy10/IG/
CUDA_VISIBLE_DEVICES=1 python main_weight_fsrcnn.py --model fsrcnn --scale 2 --batch_size 16 --patch_size 128 --print_every 10 --un_factor '10' --lr 2e-4 --decay_type 'step' --save ../experiment/FSRCNN/FSRCNN_x2_IG_filter_un10_RF0.1_Sy10 --remain_factor 0.1  --record_filters_folder ../experiment/FSRCNN/FSRCNN_x2_IG_un10_RF0.1_Sy10/IG/

# EDSR
CUDA_VISIBLE_DEVICES=2 python main_weight_edsr.py --model edsr_baseline --scale 4 --patch_size 128 --batch_size 16 --un_factor '10' --lr 2e-4 --decay_type 'step' --save ../experiment/EDSR/EDSR_x4_IG_un10_RF0.1_Sy1  --remain_factor 0.1 --record_filters_folder ../experiment/EDSR/EDSR_x4_IG_filter_un10_RF0.1_Sy1/IG/
CUDA_VISIBLE_DEVICES=2 python main_weight_edsr.py --model edsr_baseline --scale 3 --patch_size 129 --batch_size 16 --print_every 10 --un_factor '10' --lr 2e-4 --decay_type 'step' --save ../experiment/EDSR/EDSR_x3_IG_un10_RF0.1_Sy1 --remain_factor 0.1 --record_filters_folder ../experiment/EDSR/EDSR_x3_IG_un10_RF0.1_Sy1/IG/
CUDA_VISIBLE_DEVICES=2 python main_weight_edsr.py --model edsr_baseline --scale 2 --patch_size 128 --batch_size 16 --print_every 10 --un_factor '10' --lr 2e-4 --decay_type 'step' --save ../experiment/EDSR/EDSR_x2_IG_un10_RF0.1_Sy1 --remain_factor 0.1 --record_filters_folder ../experiment/EDSR/EDSR_x2_IG_un10_RF0.1_Sy1/IG/

# other
CUDA_VISIBLE_DEVICES=3 python main_weight_swinir.py --model swinir --scale 4 --batch_size 16 --print_every 10 --un_factor '10' --lr 2e-4 --decay_type 'step' --save ../experiment/SWINIR/bfig_Sy1  --remain_factor 0.1 --record_filters_folder ../experiment/SWINIR/bfig_Sy1/IG/
CUDA_VISIBLE_DEVICES=2 python main_weight_nlsn.py --model nlsn --scale 4 --batch_size 16 --print_every 10 --un_factor '10' --lr 2e-4 --decay_type 'step' --save ../experiment/NLSN/bfig_Sy1  --remain_factor 0.1 --record_filters_folder ../experiment/NLSN/bfig_Sy1/IG/