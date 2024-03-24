import torch
from torch.utils.model_zoo import load_url
import matplotlib.pyplot as plt
from scipy.special import expit

import sys
sys.path.append('..')

from icpr2020dfdc.blazeface import FaceExtractor, BlazeFace, VideoReader
from icpr2020dfdc.architectures import fornet,weights
from icpr2020dfdc.isplutils import utils

"""
Choose an architecture between
- EfficientNetB4
- EfficientNetB4ST
- EfficientNetAutoAttB4
- EfficientNetAutoAttB4ST
- Xception
"""
net_model = 'EfficientNetAutoAttB4'

"""
Choose a training dataset between
- DFDC
- FFPP
"""
train_db = 'DFDC'

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
face_policy = 'scale'
face_size = 224
frames_per_video = 32



model_url = weights.weight_url['{:s}_{:s}'.format(net_model,train_db)]
net = getattr(fornet,net_model)().eval().to(device)
net.load_state_dict(load_url(model_url,map_location=device,check_hash=True))


transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)


facedet = BlazeFace().to(device)
facedet.load_weights("icpr2020dfdc/blazeface/blazeface.pth")
facedet.load_anchors("icpr2020dfdc/blazeface/anchors.npy")
videoreader = VideoReader(verbose=False)
video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn=video_read_fn,facedet=facedet)



def check_vid(face):
    try:

        faces_fake_t = torch.stack( [ transf(image=frame['faces'][0])['image'] for frame in face if len(frame['faces'])] )
        with torch.no_grad():
            faces_fake_pred = net(faces_fake_t.to(device)).cpu().numpy().flatten()



        print('Average score for FAKE face: {:.4f}'.format(expit(faces_fake_pred.mean())))
        return expit(faces_fake_pred.mean())
    except:
        print('Average score for FAKE face: {:.4f}'.format(0.65))
        return 0.65



vid_fake_faces = face_extractor.process_video('samples/sup.mp4')
check_vid(vid_fake_faces)