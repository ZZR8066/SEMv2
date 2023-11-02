#! /bin/bash

dlp submit \
     -a jszhang6 \
     -n Train \
     -d "SEMv2-M17-TAL" \
     -i reg.deeplearning.cn/dlaas/cv_dist:0.1 \
     -e train.sh \
     -l train_dlp.log \
     -o train_dlp.err \
     --useGpu \
     -g 2 \
     -t PtJob \
     --useDist \
     -w 1 \
     -k TeslaA100-PCIE-48GB \
     -r "dlp3-superbrain-pretrain-reserved"
     # -k TeslaV100-PCIE-32GB \
     # -r "dlp3-superbrain-reserved"