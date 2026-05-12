#!/usr/bin/env python3
"""
Falcon Pitch Calibration Tool

Clica nos landmarks do campo em qualquer frame do vídeo — cada ponto pode estar
numa frame diferente (ex: cantos só visíveis quando a câmara faz pan).

O sistema usa optical flow para ajustar todos os cliques ao mesmo referencial
(frame 0), por isso a homografia final é sempre consistente independentemente
de em que frame cada ponto foi marcado.

Uso:
    python calibrate.py --input "input_videos/campo_vazio.mp4"

Controlos:
    Click       — Coloca o landmark atual
    N / →       — Próximo landmark (sem clicar)
    B / ←       — Landmark anterior
    U           — Desfaz o ponto atual
    → / ←       — Avança / recua 1 frame
    F / Espaço  — Avança 30 frames
    R           — Recua 30 frames
    G           — Vai para uma frame específica (pede número)
    S           — Guarda a calibração (mínimo 4 pontos)
    Q / Esc     — Sai sem guardar
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np

WINDOW = "Falcon Calibration  (S=guardar  Q=sair)"
CALIBRATION_PATH = "calibration/calibration.json"

LANDMARKS = [
    (0,  "Canto sup. esq. do campo"),
    (5,  "Canto inf. esq. do campo"),
    (24, "Canto sup. dir. do campo"),
    (29, "Canto inf. dir. do campo"),
    (13, "Linha de meio-campo — topo"),
    (16, "Linha de meio-campo — baixo"),
    (1,  "Área esq. — canto sup. esq."),
    (4,  "Área esq. — canto inf. esq."),
    (9,  "Área esq. — canto sup. dir."),
    (12, "Área esq. — canto inf. dir."),
    (25, "Área dir. — canto sup. dir."),
    (28, "Área dir. — canto inf. dir."),
    (17, "Área dir. — canto sup. esq."),
    (20, "Área dir. — canto inf. esq."),
]

WORLD_COORDS = {
    0:  (0,     0),      5:  (0,     7000),
    24: (12000, 0),      29: (12000, 7000),
    13: (6000,  0),      16: (6000,  7000),
    1:  (0,     1450),   4:  (0,     5550),
    9:  (2015,  1450),   12: (2015,  5550),
    25: (12000, 1450),   28: (12000, 5550),
    17: (9985,  1450),   20: (9985,  5550),
}

COL_DONE    = (0, 220, 0)
COL_CURRENT = (0, 180, 255)
COL_PENDING = (120, 120, 120)
COL_TEXT    = (255, 255, 255)
COL_BG      = (30, 30, 30)
PANEL_W     = 410


# ── optical flow para compensar movimento de câmara ───────────────────────────

def _flow_params():
    lk = dict(winSize=(21, 21), maxLevel=3,
               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
    feat = dict(maxCorners=200, qualityLevel=0.01, minDistance=5, blockSize=7)
    return lk, feat

def precompute_movements(frames: list[np.ndarray]) -> np.ndarray:
    """
    Calcula o deslocamento CUMULATIVO de câmara de cada frame relativamente
    à frame 0, usando Lucas-Kanade optical flow com mediana robusta.

    Retorna array shape [N, 2] onde movements[i] = (cum_dx, cum_dy) desde frame 0.
    """
    lk_params, feat_params = _flow_params()
    N = len(frames)
    movements = np.zeros((N, 2), dtype=np.float32)

    def border_mask(gray):
        h, w = gray.shape
        mask = np.zeros_like(gray)
        mask[0:h//7, :]      = 1
        mask[h-h//7:, :]     = 1
        mask[:, 0:w//7]      = 1
        mask[:, w-w//7:]     = 1
        return mask

    old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    mask = border_mask(old_gray)
    feat_params_masked = {**feat_params, "mask": mask}
    old_pts = cv2.goodFeaturesToTrack(old_gray, **feat_params_masked)

    print("A calcular movimento de câmara entre frames...", end="", flush=True)

    for i in range(1, N):
        if i % 50 == 0:
            print(f"\r  Optical flow: {i}/{N} frames processadas...   ", end="", flush=True)

        new_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        if old_pts is None or len(old_pts) < 4:
            old_pts = cv2.goodFeaturesToTrack(old_gray, **feat_params_masked)
            movements[i] = movements[i - 1]
            old_gray = new_gray
            continue

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            old_gray, new_gray, old_pts, None, **lk_params)

        if new_pts is None or status is None:
            movements[i] = movements[i - 1]
            old_gray = new_gray
            old_pts = cv2.goodFeaturesToTrack(old_gray, **feat_params_masked)
            continue

        good_new = new_pts[status.ravel() == 1]
        good_old = old_pts[status.ravel() == 1]

        if len(good_new) < 4:
            movements[i] = movements[i - 1]
        else:
            delta = good_new.reshape(-1, 2) - good_old.reshape(-1, 2)
            dx = float(np.median(delta[:, 0]))
            dy = float(np.median(delta[:, 1]))
            movements[i] = movements[i - 1] + np.array([dx, dy])

        old_gray = new_gray
        if len(good_new) >= 4:
            old_pts = good_new.reshape(-1, 1, 2).astype(np.float32)
        else:
            old_pts = cv2.goodFeaturesToTrack(old_gray, **feat_params_masked)

    print(f"\r  Optical flow concluído ({N} frames).                    ")
    return movements


# ── UI helpers ─────────────────────────────────────────────────────────────────

def draw_panel(panel: np.ndarray, current_idx: int, placed: dict) -> None:
    panel[:] = COL_BG
    cv2.putText(panel, "LANDMARKS DO CAMPO", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, COL_TEXT, 1)
    cv2.line(panel, (10, 38), (PANEL_W - 10, 38), (80, 80, 80), 1)

    for i, (vidx, name) in enumerate(LANDMARKS):
        y = 60 + i * 34
        done = vidx in placed

        if i == current_idx:
            cv2.rectangle(panel, (4, y - 18), (PANEL_W - 4, y + 14),
                          (55, 55, 110), -1)
            col, prefix = COL_CURRENT, ">"
        elif done:
            col, prefix = COL_DONE, "v"
        else:
            col, prefix = COL_PENDING, " "

        label = f" {prefix} {i+1:2d}. {name}"
        cv2.putText(panel, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1)

        if done:
            px, py, fn = placed[vidx]
            cv2.putText(panel, f"      px=({px},{py})  fr={fn}",
                        (8, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34, COL_DONE, 1)

    footer = panel.shape[0] - 145
    cv2.line(panel, (10, footer), (PANEL_W - 10, footer), (80, 80, 80), 1)
    for j, line in enumerate([
        "Click   colocar ponto",
        "N / ->  proximo",
        "B / <-  anterior",
        "U       desfazer",
        "F/Spc   +30 frames",
        "R       -30 frames",
        "G       ir para frame",
        "S       guardar",
        "Q/Esc   sair",
    ]):
        cv2.putText(panel, line, (10, footer + 16 + j * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    n = len(placed)
    col_n = COL_DONE if n >= 4 else (0, 120, 220)
    cv2.putText(panel, f"{n}/{len(LANDMARKS)} pontos colocados",
                (10, panel.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, col_n, 1)


def draw_overlay(frame: np.ndarray, current_idx: int, placed: dict,
                 frame_idx: int, total: int, movements: np.ndarray) -> np.ndarray:
    out = frame.copy()

    # Pontos já colocados — reposicionados para a frame atual
    for vidx, (px, py, fn) in placed.items():
        # Shift do ponto para a frame atual: pixel_ref → pixel_atual
        ref_cum = movements[fn]          # deslocamento cumulativo na frame em que foi clicado
        cur_cum = movements[frame_idx]   # deslocamento cumulativo na frame atual
        delta = cur_cum - ref_cum
        cx = int(px + delta[0])
        cy = int(py + delta[1])

        cv2.circle(out, (cx, cy), 9, COL_DONE, -1)
        cv2.circle(out, (cx, cy), 9, (0, 0, 0), 2)
        lm_idx = next((i for i, (v, _) in enumerate(LANDMARKS) if v == vidx), -1)
        if lm_idx >= 0:
            cv2.putText(out, str(lm_idx + 1), (cx + 11, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, COL_DONE, 2)

    # Instrução do ponto atual
    _, name = LANDMARKS[current_idx]
    msg = f"Clica em: {current_idx+1}. {name}"
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)
    cv2.rectangle(out, (6, 6), (tw + 14, th + 18), (0, 0, 0), -1)
    cv2.putText(out, msg, (10, th + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, COL_CURRENT, 2)

    # Deslocamento de câmara acumulado (informação de debug)
    cum = movements[frame_idx]
    cam_txt = f"Cam offset: ({cum[0]:.1f}, {cum[1]:.1f}) px"
    cv2.putText(out, cam_txt, (10, out.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Frame counter
    cv2.putText(out, f"Frame {frame_idx+1}/{total}",
                (10, out.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
    return out


# ── main ───────────────────────────────────────────────────────────────────────

def load_frames(video_path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    if not frames:
        print(f"Erro: não foi possível abrir '{video_path}'")
        sys.exit(1)
    return frames


def run(video_path: str) -> None:
    os.makedirs("calibration", exist_ok=True)

    print(f"A carregar '{video_path}'...")
    frames = load_frames(video_path)
    total = len(frames)
    print(f"{total} frames carregadas.")

    # Pré-computar movimentos de câmara para todas as frames
    movements = precompute_movements(frames)

    frame_idx  = 0
    current_lm = 0
    # placed: vertex_idx → (pixel_x, pixel_y, frame_num)
    placed: dict[int, tuple[int, int, int]] = {}
    click_pos: list[tuple[int, int] | None] = [None]

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            frame_x = x - PANEL_W
            if frame_x >= 0:
                click_pos[0] = (frame_x, y)

    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        frame = frames[frame_idx]
        h = frame.shape[0]
        panel = np.zeros((h, PANEL_W, 3), dtype=np.uint8)
        draw_panel(panel, current_lm, placed)
        overlay = draw_overlay(frame, current_lm, placed, frame_idx, total, movements)
        cv2.imshow(WINDOW, np.hstack([panel, overlay]))

        # Clique
        if click_pos[0] is not None:
            px, py = click_pos[0]
            click_pos[0] = None
            vidx, name = LANDMARKS[current_lm]
            placed[vidx] = (px, py, frame_idx)
            print(f"  [{current_lm+1}] {name} → pixel ({px},{py}) na frame {frame_idx}")
            if current_lm < len(LANDMARKS) - 1:
                current_lm += 1

        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key in (ord('n'), 83):   # N ou →
            current_lm = min(current_lm + 1, len(LANDMARKS) - 1)
        elif key in (ord('b'), 81):   # B ou ←
            current_lm = max(current_lm - 1, 0)
        elif key == ord('u'):
            vidx, _ = LANDMARKS[current_lm]
            if vidx in placed:
                del placed[vidx]
                print(f"  Ponto {current_lm+1} removido.")
            elif current_lm > 0:
                current_lm -= 1
                vidx, _ = LANDMARKS[current_lm]
                if vidx in placed:
                    del placed[vidx]
                    print(f"  Ponto {current_lm+1} removido.")
        elif key in (ord('f'), 32):   # F ou Espaço
            frame_idx = min(frame_idx + 30, total - 1)
        elif key == ord('r'):
            frame_idx = max(frame_idx - 30, 0)
        elif key == 82:               # seta cima (OpenCV)
            frame_idx = min(frame_idx + 1, total - 1)
        elif key == 84:               # seta baixo
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord('g'):
            cv2.destroyWindow(WINDOW)
            raw = input(f"Ir para frame (0-{total-1}): ").strip()
            if raw.isdigit():
                frame_idx = max(0, min(int(raw), total - 1))
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(WINDOW, on_mouse)
        elif key == ord('s'):
            if len(placed) < 4:
                print("Precisa de pelo menos 4 pontos.")
                continue

            # Ajustar todos os pontos ao referencial da frame 0
            # pixel_ref = pixel_clicado - cumulative_movement[frame_clicado]
            # (assim ficam em coordenadas da frame 0)
            pixel_pts = []
            world_pts = []
            for vidx, (px, py, fn) in placed.items():
                cum = movements[fn]
                adj_x = float(px) - float(cum[0])
                adj_y = float(py) - float(cum[1])
                pixel_pts.append([adj_x, adj_y])
                wx, wy = WORLD_COORDS[vidx]
                world_pts.append([float(wx), float(wy)])

            data = {
                "pixel_points":  pixel_pts,
                "world_points":  world_pts,
                "n_points":      len(placed),
                "video":         video_path,
                "note": ("Pixel points are expressed in frame-0 coordinate space. "
                         "Adjust by cumulative camera movement before use."),
            }
            with open(CALIBRATION_PATH, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\nCalibração guardada em '{CALIBRATION_PATH}' com {len(placed)} pontos.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()
    print("Calibração não foi guardada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Falcon Pitch Calibration Tool")
    parser.add_argument("--input", required=True,
                        help="Caminho para o vídeo de referência")
    args = parser.parse_args()
    run(args.input)
