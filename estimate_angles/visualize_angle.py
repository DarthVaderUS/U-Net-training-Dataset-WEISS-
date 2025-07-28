import sys, cv2, numpy as np, matplotlib.pyplot as plt
from image_utils import load_mask, extract_skeleton, get_ordered_skeleton_path, smooth_path, compute_total_turn_angle

def visualize(p):
    mask=load_mask(p)
    sk=extract_skeleton(mask)
    raw=get_ordered_skeleton_path(sk)
    path=smooth_path(raw)
    total=compute_total_turn_angle(path)
    img=cv2.cvtColor(mask*255,cv2.COLOR_GRAY2BGR)
    for (r,c) in path.astype(int):
        img[r,c]=(0,0,255)
    s=tuple(path[0][::-1].astype(int)); e=tuple(path[-1][::-1].astype(int))
    cv2.circle(img,s,4,(0,255,0),-1); cv2.circle(img,e,4,(255,0,0),-1)
    plt.figure(figsize=(5,5)); plt.imshow(img[...,::-1])
    plt.title(f"Total turn: {total:.2f}°"); plt.axis('off'); plt.show()

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("Usage: python visualize_angle.py <mask.png>"); sys.exit(1)
    visualize(sys.argv[1])
