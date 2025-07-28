import sys
from image_utils import estimate_turn_angle

def main():
    if len(sys.argv)!=3:
        print("Usage: python estimate_angle.py <mask1.png> <mask2.png>")
        sys.exit(1)
    m1,m2 = sys.argv[1],sys.argv[2]
    try:
        a1 = estimate_turn_angle(m1)
        a2 = estimate_turn_angle(m2)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    diff = a2 - a1
    print(f"Angle1: {a1:.2f}°")
    print(f"Angle2: {a2:.2f}°")
    print(f"Angle Difference: {diff:.2f}°")

if __name__=="__main__":
    main()
