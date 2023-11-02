#! /bin/bash

dlp submit \
     -a jszhang6 \
     -n Valid \
     -d "SEMv2-C1-TAL" \
     -i reg.deeplearning.cn/dlaas/cv_dist:0.1 \
     -e valid.sh \
     -l valid_dlp.log \
     -o valid_dlp.err \
     --useGpu \
     -g 1 \
     -k TeslaV100-PCIE-32GB \
     -t PtJob \
     --useDist \
     -w 1 \
     -r "dlp3-superbrain-reserved"
     # -r "dlp3-superbrain-pretrain-reserved"