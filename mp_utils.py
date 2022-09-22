import pickle
import cv2
import numpy
from mediapipe import solutions
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

def get_mp_pose_proccessor():
    return solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation=True)

def get_simplified_pose(img, mp_pose_proccessor, target="recording", skip_incomplete=False):
    lm = (mp_pose_proccessor.process(img)).pose_landmarks
    x = NormalizedLandmarkList
    if (lm is None):
        return None

    if (skip_incomplete==True):
        for node in lm.landmark:
            if (node.x<0 or node.x>1 or node.y<0 or node.y>1):
                return None

    nodes = []
    nodes.append(((lm.landmark[7].x+lm.landmark[8].x)/2, (lm.landmark[7].y+lm.landmark[8].y)/2, (lm.landmark[7].z+lm.landmark[8].z)/2))
    nodes.append(((lm.landmark[11].x+lm.landmark[12].x)/2, (lm.landmark[11].y+lm.landmark[12].y)/2, (lm.landmark[11].z+lm.landmark[12].z)/2))
    nodes.append((lm.landmark[13].x, lm.landmark[13].y, lm.landmark[13].z))
    nodes.append((lm.landmark[15].x, lm.landmark[15].y, lm.landmark[15].z))
    nodes.append((lm.landmark[14].x, lm.landmark[14].y, lm.landmark[14].z))
    nodes.append((lm.landmark[16].x, lm.landmark[16].y, lm.landmark[16].z))
    nodes.append(((lm.landmark[23].x+lm.landmark[24].x)/2, (lm.landmark[23].y+lm.landmark[24].y)/2, (lm.landmark[23].z+lm.landmark[24].z)/2))
    nodes.append((lm.landmark[25].x, lm.landmark[25].y, lm.landmark[25].z))
    nodes.append((lm.landmark[27].x, lm.landmark[27].y, lm.landmark[27].z))
    nodes.append((lm.landmark[26].x, lm.landmark[26].y, lm.landmark[26].z))
    nodes.append((lm.landmark[28].x, lm.landmark[28].y, lm.landmark[28].z))
    if (target=="recording"):
        return nodes

    elif (target=="ploting"):
        while (len(lm.landmark)>len(nodes)):
            lm.landmark.pop(-1)
        for i in range(len(nodes)):
            lm.landmark[i].x = nodes[i][0]
            lm.landmark[i].y = nodes[i][1]
            lm.landmark[i].z = nodes[i][2]
            lm.landmark[i].visibility = 0.999
        return lm

def get_simplified_pose_connections():
    return ((0, 1), (1, 2), (1, 4), (1, 6), (2, 3), (4, 5), (6, 7), (6, 9), (7, 8), (9, 10))

def get_simplified_pose_landmarks_style():
    n = solutions.drawing_utils.DrawingSpec()
    n.circle_radius = 2
    n.thickness = 2
    map = {}
    for i in range(11):
        map[i] = n
    return map

def get_pose_segmentation_mask(img, mp_pose_proccessor, target="recording"):
    result = mp_pose_proccessor.process(img)
    if (result.segmentation_mask is None):
        return None
    else:
        #using contour to improve storing effiefficiency
        mask = (result.segmentation_mask>0.1).astype(numpy.uint8)
        (cts, hier) = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        ct_area_size = []
        holes = []
        for i in range(len(cts)):
            if (hier[0][i][3]>=0):
                holes.append(cts[i])
                continue
            ct_area_size.append((i, cv2.contourArea(cts[i])))

        #only save holes & contours that are big enough (reduce noise)
        ct_area_size = sorted(ct_area_size, key=lambda pair:pair[1], reverse=True)

        if (target=="recording"):
            ct_lst = []
            for i in range(len(ct_area_size)):
                if (ct_area_size[i][1]<ct_area_size[0][1]/100):
                    break
                ct_lst.append(cts[ct_area_size[i][0]].tolist())
            #from this idx, contours are holes
            hole_idx = len(ct_lst)
            for i in range(len(holes)):
                ct_lst.append(holes[i].tolist())
            
            return [ct_lst, hole_idx]

        elif (target=="ploting"):
            canvas = numpy.zeros(mask.shape, mask.dtype)
            for i in range(len(ct_area_size)):
                if (ct_area_size[i][1]<ct_area_size[0][1]/100):
                    break
                cv2.drawContours(canvas, cts, ct_area_size[i][0], 255, cv2.FILLED)
            for i in range(len(holes)):
                cv2.drawContours(canvas, holes, i, 0, cv2.FILLED)

            return cv2.cvtColor(canvas.astype(numpy.float32), cv2.COLOR_GRAY2BGR)

def get_landmark_template():
    with open("./assets/landmark_template", "rb") as fin:
        return pickle.load(fin)

def generate_landmark(landmark_template, torch_tensor):
    coord_lst = torch_tensor.squeeze().tolist()
    for i in range(len(landmark_template.landmark)):
        landmark_template.landmark[i].x = coord_lst[0+i*3]
        landmark_template.landmark[i].y = coord_lst[1+i*3]
        landmark_template.landmark[i].z = coord_lst[2+i*3]
    return landmark_template

if __name__=="__main__":
    video_stream = cv2.VideoCapture(0)
    proccessor = get_mp_pose_proccessor()
    connection = get_simplified_pose_connections()
    pls = get_simplified_pose_landmarks_style()

    while True:
        (ret, frame) = video_stream.read()
        if frame is None:
            continue
        frame.flags.writeable = False
        lm = get_simplified_pose(frame, proccessor, target="ploting", skip_incomplete=False)
        frame.flags.writeable = True

        #Skeleton mask
        frame0 = numpy.zeros(frame.shape, dtype=numpy.uint8)
        solutions.drawing_utils.draw_landmarks(frame0, lm, connection, pls)

        #Binary mask
        frame1 = get_pose_segmentation_mask(frame, proccessor, "ploting")
        if (frame0 is not None and frame1 is not None):
            cv2.imshow("Skeleton mask", cv2.flip(frame0, 1))
            cv2.imshow("Binary mask", cv2.flip(frame1, 1))
        if (cv2.waitKey(1)&0xFF==27):
            break
    
    video_stream.release()