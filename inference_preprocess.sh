python Phase2/inference.py \
--input_folder inputs/PhaseThreeData \
--output_folder Phase2/outputs \
--kernel_model_path pretrained/kernel_model.pth \
--restore_model_path pretrained/restore_model.pth

python HAT/hat/test.py \
-opt HAT/options/test/HAT_SRx2.yml