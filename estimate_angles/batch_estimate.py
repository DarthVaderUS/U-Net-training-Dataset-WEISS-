import os, csv
from image_utils import estimate_turn_angle

def batch_estimate(folder='dataset/masks', out_csv='angle_results.csv'):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith('.png'))
    rows=[]
    for i in range(len(files)-1):
        f1,f2 = files[i],files[i+1]
        p1,p2 = os.path.join(folder,f1),os.path.join(folder,f2)
        try:
            a1,a2 = estimate_turn_angle(p1), estimate_turn_angle(p2)
            diff = a2 - a1
            rows.append([f1,f2,f"{a1:.2f}",f"{a2:.2f}",f"{diff:.2f}"])
            print(f"{f1} vs {f2} → Δ {diff:.2f}°")
        except Exception as e:
            print(f"Error {f1} vs {f2}: {e}")
    with open(out_csv,'w',newline='') as fp:
        w=csv.writer(fp)
        w.writerow(['Image1','Image2','Angle1_deg','Angle2_deg','Delta_deg'])
        w.writerows(rows)
    print(f"\nSaved to {out_csv}")

if __name__=="__main__":
    batch_estimate()
