import torch
# model = torch.load('./output/pssm_trembl_01/model_1.pth', map_location='cpu')
# torch.save(model.module, './output/pssm_trembl_01/model_1_cpu.pth')
model = torch.load('./output/vit_15/model_8.pth', map_location='cpu')
# torch.save(model.module, './output/vit_12/model_9_single.pth')
torch.save(model, './output/vit_15/model_8_single.pth')

# python pssm_running.py -o ./output/pssm_trembl_01/swissprot_20180412_20210531 -ckpt ./output/pssm_trembl_01/model_0_cpu.pth -b 8 -g cpu -i ./Dataset/pssm_data/SwissProt_20180412_20210531_pssm.npz
# python ViT_running.py -i ./Dataset/analysis_seqs/swissprot_20180412_20210531.fa -ckpt ./output/vit_14/model_17_single.pth -o ./output/vit_14/swissprot_20180412_20210531 -b 256 -g cuda:1
# python ViT_running.py -i ./Dataset/analysis_seqs/swissprot_20180412_20210914.fa -ckpt ./output/vit_15/model_8_single.pth -o ./output/vit_15/swissprot_20180412_20210914 -b 256 -g cuda:2