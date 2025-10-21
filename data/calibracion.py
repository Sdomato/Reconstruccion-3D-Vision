# src/calib/calibrate_from_images.py
import argparse, glob, os, pickle
import cv2 as cv
import numpy as np

COMMON_SIZES = [(9,6),(8,6),(7,6),(7,5)]  # (cols, rows) de esquinas internas

def detect_size_auto(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    for cols, rows in COMMON_SIZES:
        ok, _ = cv.findChessboardCorners(img, (cols, rows), flags=cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            return (cols, rows)
    raise RuntimeError("No pude detectar el checkerboard con tamaños comunes. Pasá --cols y --rows explícitos.")

def collect_points(left_paths, right_paths, pattern_size, square_size):
    cols, rows = pattern_size
    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2) * square_size

    objpoints = []
    imgpointsL, imgpointsR = [], []
    crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

    for lp, rp in zip(left_paths, right_paths):
        imgL = cv.imread(lp, cv.IMREAD_GRAYSCALE)
        imgR = cv.imread(rp, cv.IMREAD_GRAYSCALE)
        okL, cornersL = cv.findChessboardCorners(imgL, (cols, rows), flags=cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_NORMALIZE_IMAGE)
        okR, cornersR = cv.findChessboardCorners(imgR, (cols, rows), flags=cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_NORMALIZE_IMAGE)
        if not (okL and okR):
            print(f"[WARN] saltando par por no detectar corners: {os.path.basename(lp)} | {os.path.basename(rp)}")
            continue
        cornersL = cv.cornerSubPix(imgL, cornersL, (11,11), (-1,-1), crit)
        cornersR = cv.cornerSubPix(imgR, cornersR, (11,11), (-1,-1), crit)
        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)
    return objpoints, imgpointsL, imgpointsR, imgL.shape[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Carpeta con imágenes calib_*. Ej: datasets/stereo_budha_board/calib')
    ap.add_argument('--out', default='data', help='Carpeta de salida para .pkl')
    ap.add_argument('--cols', type=int)
    ap.add_argument('--rows', type=int)
    ap.add_argument('--square', type=float, default=20.0, help='Tamaño del cuadrado en mm (ajusta para escala real)')
    ap.add_argument('--auto', action='store_true', help='Detecta automáticamente cols/rows en la primera imagen izquierda')
    args = ap.parse_args()

    left_paths = sorted(glob.glob(os.path.join(args.data, 'calib_left_*.jpg')))
    right_paths = sorted(glob.glob(os.path.join(args.data, 'calib_right_*.jpg')))
    assert len(left_paths) == len(right_paths) and len(left_paths)>0, 'No se encontraron pares L/R.'

    if args.auto:
        pattern_size = detect_size_auto(left_paths[0])
        print(f"[INFO] Tamaño detectado (cols, rows): {pattern_size}")
    else:
        assert args.cols and args.rows, 'Especificá --cols y --rows o usa --auto.'
        pattern_size = (args.cols, args.rows)

    objpoints, imgpointsL, imgpointsR, imsize = collect_points(left_paths, right_paths, pattern_size, args.square)
    print(f"[INFO] Pares válidos: {len(objpoints)} / {len(left_paths)}")

    # Calibración individual
    retL, K1, D1, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, imsize, None, None)
    retR, K2, D2, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, imsize, None, None)

    # Calibración estéreo
    flags = cv.CALIB_FIX_INTRINSIC
    criteria = (cv.TERM_CRITERIA_MAX_ITER+cv.TERM_CRITERIA_EPS, 100, 1e-5)
    rms, K1, D1, K2, D2, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR, K1, D1, K2, D2, imsize,
        criteria=criteria, flags=flags
    )
    print(f"[INFO] Stereo RMS: {rms:.4f}")

    # Rectificación
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(K1, D1, K2, D2, imsize, R, T, flags=cv.CALIB_ZERO_DISPARITY)
    left_map_x, left_map_y = cv.initUndistortRectifyMap(K1, D1, R1, P1, imsize, cv.CV_32FC1)
    right_map_x, right_map_y = cv.initUndistortRectifyMap(K2, D2, R2, P2, imsize, cv.CV_32FC1)

    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.out, 'stereo_calibration.pkl'), 'wb') as f:
        pickle.dump({
            'K1':K1, 'D1':D1, 'K2':K2, 'D2':D2, 'R':R, 'T':T, 'E':E, 'F':F,
            'R1':R1, 'R2':R2, 'P1':P1, 'P2':P2, 'Q':Q,
            'pattern_size':pattern_size, 'square_size_mm':args.square
        }, f)
    with open(os.path.join(args.out, 'stereo_maps.pkl'), 'wb') as f:
        pickle.dump({
            'left_map_x':left_map_x, 'left_map_y':left_map_y,
            'right_map_x':right_map_x, 'right_map_y':right_map_y,
            'Q':Q
        }, f)
    print(f"[OK] Guardado en {args.out}/stereo_calibration.pkl y {args.out}/stereo_maps.pkl")

if __name__ == '__main__':
    main()