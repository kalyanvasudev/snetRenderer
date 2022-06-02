import submitit
import os
import os.path as osp
os.environ["MKL_THREADING_LAYER"] = "GNU"
from renderer.BlenderRenderer import Renderer
import copy
import numpy as np
import copy
import startup
#from Pose import Pose
import ast
import numpy as np
import scipy.io as sio
import skimage.io
import math
import random

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
import time
#import cPickle as pickle
import pickle 
import errno
from tqdm import tqdm
_curr_path = osp.dirname(osp.abspath(__file__))
class shapenetRenderer():
    def __init__(self, numPoses=50, useV1=True, synsets=[], synsetModels=[]):
        self.numPoses = numPoses

        self.useV1 = useV1
        self.synsets = synsets
        self.synsetModels = synsetModels
        self.config = startup.params()
        #print self.synsetModels[0][0]
        #self.renderer = Renderer([self.synsetModels[0][0]],self.imX, self.imY)
        self.renderer = Renderer([self.synsetModels[0][0]])

    def renderAllSynsets(self, minS=0, maxS=None):
        if(maxS == None):
            maxS = len(self.synsets)
            
        for sId in range(minS, maxS):
            synset = self.synsets[sId]
            print(synset)
            for mId in tqdm(range(len(self.synsetModels[sId]))):
            #for mId in range(2):
                print(mId)
                mName = self._loadNewModel(sId, mId)
                renderDir = osp.join(self.config['renderPrecomputeDir'], synset, mName)
                mkdir_p(renderDir)
                poseSamples = self._randomPoseSamples()
                cam_info = np.array(poseSamples)
                infoFile = osp.join(renderDir,'poseInfo.pickle')
                np.save(osp.join(renderDir, "cam_info"), cam_info)

                self.renderer.renderViews(poseSamples, renderDir)

    def _save_poses(self):
        pass

    def _loadNewModel(self, synsetId, modelId):
        mPath = self.synsetModels[synsetId][modelId]
        if(self.useV1):
            mName = (mPath.split('/'))[-2]
        else:
            mName = (mPath.split('/'))[-3]
        self.renderer.reInit([mPath])
        return mName

    def _randomPoseSamples(self):
        poseSamples = []
        for n in range(self.numPoses):
            azim = -180 + 360 * random.random()
            elev = 70 * random.random() # for encoder
            #elev = -90  + 180 * random.random()
            theta = 0.0
            dist = 1.8 + 0.7 * random.random()
            poseSample = [azim, elev, theta, dist]
            poseSamples.append(poseSample)
        return poseSamples



def initModelInfo(useV1=True):
    config = startup.params()
    # synsets = ['02858304','02924116','03790512','04468005', '02992529','02843684', '02954340']
    # synsets = ['02933112','03636649','04090263','04379243','04530566',\
    #           '02828884','03211117','03691459','04256520','04401088', '02691156','02958343', '03001627']
    
    synsets = ['02747177','02773838','02801938','02808440','02818832','02834778','02871439','02876657','02880940','02942699',
            '02946921','03085013','03046257','03207941','03261776','03325088','03337140',
               '03467517','03513137','03593526','03624134','03642806','03710193','03759954','03761084',
               '03797390','03928116','03938244','03948459','03991062','04004475',
               '04074963','04099429','04225987','04330267',
               '04460130','04554684']

    if(useV1):
        #synsets = [f[0:-4] for f in os.listdir(config['shapenetDir']) if f.endswith('.csv')]
        synsets.sort()
        synsetModels = [[osp.join(config['shapenetDir'],s,f,'model.obj') for f in os.listdir(osp.join(config['shapenetDir'],s)) if len(f) > 3] for s in synsets]
    else:
        synsets = [f for f in os.listdir(config['shapenetDir']) if len(f)>3]
        synsets.sort()
        synsetModels = [[osp.join(config['shapenetDir'],s,f,'models','model_normalized.obj') for f in os.listdir(osp.join(config['shapenetDir'],s)) if len(f) > 3] for s in synsets]
    
    return synsets, synsetModels


# ## Debugg
# if __name__ == "__main__":
#     synsets, synsetModels = initModelInfo()
#     renderer = shapenetRenderer(numPoses=50, useV1=True, synsets=synsets, synsetModels=synsetModels)
#     renderer.renderAllSynsets(minS=0, maxS=5)

def render_v1_images(synset, models):

    renderer = shapenetRenderer(numPoses=50, useV1=True, synsets=[synset], synsetModels=[models])
    renderer.renderAllSynsets(minS=0, maxS=1)


# render on cluster
if __name__ == "__main__":
    batch_size = 250
    synsets, synsetModels = initModelInfo()
    for sId in range(len(synsets)):
        synset = synsets[sId]
        models = synsetModels[sId]
        ii = 0
        while ii < len(models):
            temp_models = copy.deepcopy(models[ii: ii+ batch_size])

            submitit_dir = osp.join(
                _curr_path,
                "cachedir",
                "submitit_" + str(synset) + '_' + str(ii),
            )
            executor = submitit.AutoExecutor(folder=submitit_dir)

            job_kwargs = {
                "timeout_min": 60 * 10,
                "name":  str(ii) + '_' + str(synset),
                "slurm_partition": 'learnfair',
                "gpus_per_node": 0,
                #"tasks_per_node": 1,  # one task per GPU
                "cpus_per_task": 5,
                "nodes": 1,
            }
            executor.update_parameters(**job_kwargs)
            job = executor.submit(render_v1_images, synsets[sId], temp_models)
            print(
                "Submitit Job ID: ",
                job.job_id,
                "ii: ",
                ii,
                "total", len(models),
                "Synset: ",
                synset,
            )  # ID of your job
            ii += batch_size
            time.sleep(0.2)
        
    print("Successfully launched submitit jobs")