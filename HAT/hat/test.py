# flake8: noqa
import os.path as osp
import shutil
import hat.archs
import hat.data
import hat.models
from basicsr.test import test_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
    replaces=[34,38,40,46,47]
    for replace in replaces:
        name=str(replace).zfill(3)+'.png'
        origin='inputs/PhaseThreeData/'+name
        target='HAT/results/HAT_SRx2/visualization/'+name
        shutil.copy(origin,target)
