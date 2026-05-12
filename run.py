#!/usr/bin/env python3
"""
Falcon — Ponto de entrada interativo.
Corre este script para aceder ao menu principal.

    python run.py
"""

import os
import sys
import glob
import subprocess

# ── ANSI colours ──────────────────────────────────────────────────────────────
R  = "\033[0m"       # reset
B  = "\033[1m"       # bold
DIM= "\033[2m"       # dim
CY = "\033[96m"      # cyan
GR = "\033[92m"      # green
YE = "\033[93m"      # yellow
RE = "\033[91m"      # red
BL = "\033[94m"      # blue
MA = "\033[95m"      # magenta

CALIBRATION_FILE = "calibration/calibration.json"


# ── helpers ───────────────────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def header():
    print(f"{CY}{B}")
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║          F A L C O N  —  Football  CV           ║")
    print("  ║       Análise de jogo por visão computacional    ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print(f"{R}")

def calibration_status() -> str:
    if os.path.exists(CALIBRATION_FILE):
        import json
        with open(CALIBRATION_FILE) as f:
            d = json.load(f)
        return (f"{GR}✓ Calibração ativa{R}  "
                f"{DIM}({d['n_points']} pontos · {os.path.basename(d['video'])}){R}")
    return f"{YE}⚠  Sem calibração manual{R}  {DIM}(modo YOLO automático){R}"

def list_videos(folder: str = "input_videos") -> list[str]:
    patterns = ["*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.mkv"]
    videos = []
    for p in patterns:
        videos.extend(glob.glob(os.path.join(folder, p)))
    return sorted(videos)

def pick_video(prompt: str = "Escolhe o vídeo") -> str | None:
    videos = list_videos()
    if not videos:
        print(f"{RE}Nenhum vídeo encontrado em input_videos/{R}")
        print(f"{DIM}Copia o ficheiro .mp4 para a pasta input_videos/ e tenta de novo.{R}")
        input("\nEnter para continuar...")
        return None

    print(f"\n{B}{prompt}:{R}")
    for i, v in enumerate(videos, 1):
        name = os.path.basename(v)
        size = os.path.getsize(v) / (1024**2)
        print(f"  {CY}[{i}]{R} {name}  {DIM}({size:.0f} MB){R}")
    print(f"  {DIM}[0] Cancelar{R}")

    while True:
        raw = input(f"\n{B}Opção:{R} ").strip()
        if raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(videos):
            return videos[int(raw) - 1]
        print(f"{RE}Opção inválida.{R}")

def pick_stride() -> int:
    print(f"\n{B}Velocidade de processamento:{R}")
    print(f"  {CY}[1]{R} 30 fps  {DIM}(recomendado — processa metade dos frames, 2× mais rápido){R}")
    print(f"  {CY}[2]{R} 60 fps  {DIM}(máxima resolução temporal — demora o dobro){R}")
    while True:
        raw = input(f"\n{B}Opção [1]:{R} ").strip() or "1"
        if raw == "1":
            return 2
        if raw == "2":
            return 1
        print(f"{RE}Opção inválida.{R}")

def pick_device() -> str:
    print(f"\n{B}Dispositivo de inferência:{R}")
    print(f"  {CY}[1]{R} mps   {DIM}(Apple Silicon — M1/M2/M3){R}")
    print(f"  {CY}[2]{R} cuda  {DIM}(NVIDIA GPU){R}")
    print(f"  {CY}[3]{R} cpu   {DIM}(lento, sem GPU){R}")
    while True:
        raw = input(f"\n{B}Opção [1]:{R} ").strip() or "1"
        if raw == "1": return "mps"
        if raw == "2": return "cuda"
        if raw == "3": return "cpu"
        print(f"{RE}Opção inválida.{R}")

def confirm(msg: str) -> bool:
    raw = input(f"\n{B}{msg} [S/n]:{R} ").strip().lower()
    return raw in ("", "s", "sim", "y", "yes")

def run_cmd(cmd: list[str]) -> None:
    print(f"\n{DIM}$ {' '.join(cmd)}{R}\n")
    subprocess.run(cmd, check=False)


# ── ecrãs ─────────────────────────────────────────────────────────────────────

def screen_process():
    clear(); header()
    print(f"{B}── PROCESSAR JOGO ─────────────────────────────────────{R}")
    print(f"  {calibration_status()}\n")

    video_in = pick_video("Vídeo de entrada (jogo a analisar)")
    if video_in is None:
        return

    # Output name
    default_out = os.path.splitext(os.path.basename(video_in))[0] + "_output.mp4"
    raw = input(f"\n{B}Nome do ficheiro de saída [{default_out}]:{R} ").strip()
    out_name = raw or default_out
    if not out_name.lower().endswith(".mp4"):
        out_name += ".mp4"
    video_out = os.path.join("output_videos", out_name)

    stride = pick_stride()
    device = pick_device()

    # Aviso se já existem stubs (podem estar desatualizados)
    if any(os.path.exists(p) for p in ["stubs/track_stubs.pkl",
                                        "stubs/camera_movement_stub.pkl",
                                        "stubs/pitch_keypoints_stub.pkl"]):
        print(f"\n{YE}⚠  Encontrados stubs de processamento anterior em stubs/{R}")
        print(f"{DIM}   Se mudaste o vídeo ou os parâmetros, apaga-os para reprocessar.{R}")
        if confirm("Apagar stubs agora e forçar reprocessamento total?"):
            for p in glob.glob("stubs/*.pkl"):
                os.remove(p)
            print(f"{GR}Stubs apagados.{R}")

    # Resumo
    print(f"\n{B}── Resumo ──────────────────────────────────────────────{R}")
    print(f"  Entrada : {CY}{video_in}{R}")
    print(f"  Saída   : {CY}{video_out}{R}")
    fps_label = "30 fps" if stride == 2 else "60 fps"
    print(f"  FPS     : {CY}{fps_label}{R}")
    print(f"  Device  : {CY}{device}{R}")
    cal = f"{GR}ativa{R}" if os.path.exists(CALIBRATION_FILE) else f"{YE}desativada (YOLO){R}"
    print(f"  Calibr. : {cal}")

    if not confirm("Iniciar processamento?"):
        return

    os.makedirs("output_videos", exist_ok=True)
    run_cmd([
        sys.executable, "main.py",
        "--input",  video_in,
        "--output", video_out,
        "--stride", str(stride),
        "--device", device,
    ])

    print(f"\n{GR}{B}✓ Concluído!{R}  Vídeo guardado em: {CY}{video_out}{R}")
    input("\nEnter para continuar...")


def screen_calibrate():
    clear(); header()
    print(f"{B}── CALIBRAÇÃO DO CAMPO ─────────────────────────────────{R}\n")

    if os.path.exists(CALIBRATION_FILE):
        import json
        with open(CALIBRATION_FILE) as f:
            d = json.load(f)
        print(f"{GR}Calibração já existe:{R}")
        print(f"  Pontos  : {d['n_points']}")
        print(f"  Vídeo   : {os.path.basename(d['video'])}")
        print()
        if not confirm("Substituir por uma nova calibração?"):
            return

    print(f"""
{B}Como funciona:{R}
  1. Abre uma janela com o vídeo de referência
  2. No painel esquerdo vês a lista de landmarks do campo
  3. {CY}Clica no ponto correspondente no vídeo{R} para cada landmark
  4. Usa {CY}F{R} / {CY}Espaço{R} para avançar frames se necessário
  5. Quando terminares, prime {CY}S{R} para guardar

{YE}Dica:{R} Usa um vídeo do campo {B}sem jogadores{R} (ex: antes do aquecimento
  ou nos primeiros segundos de transmissão com o campo vazio).
  Mínimo {B}4 landmarks{R}, mas quanto mais melhor (idealmente 8+).
""")

    video_in = pick_video("Vídeo de referência (campo vazio ou início de transmissão)")
    if video_in is None:
        return

    if not confirm("Abrir ferramenta de calibração?"):
        return

    run_cmd([sys.executable, "calibrate.py", "--input", video_in])

    if os.path.exists(CALIBRATION_FILE):
        import json
        with open(CALIBRATION_FILE) as f:
            d = json.load(f)
        print(f"\n{GR}{B}✓ Calibração guardada com {d['n_points']} pontos!{R}")
        print(f"{DIM}  Os próximos processamentos irão usar esta calibração{R}")
        print(f"{DIM}  e saltar a deteção YOLO do campo (~40% mais rápido).{R}")
    else:
        print(f"\n{YE}Calibração não foi guardada.{R}")

    input("\nEnter para continuar...")


def screen_status():
    clear(); header()
    print(f"{B}── ESTADO DO SISTEMA ───────────────────────────────────{R}\n")

    # Calibração
    print(f"  Calibração : {calibration_status()}")

    # Stubs
    stubs = glob.glob("stubs/*.pkl")
    if stubs:
        print(f"\n  {GR}Stubs em cache:{R}")
        for s in stubs:
            size = os.path.getsize(s) / (1024**2)
            print(f"    {DIM}{os.path.basename(s)}  ({size:.1f} MB){R}")
        print(f"\n  {DIM}Os stubs aceleram runs subsequentes no mesmo vídeo.{R}")
        if confirm("\n  Apagar todos os stubs?"):
            for s in stubs:
                os.remove(s)
            print(f"  {GR}Stubs apagados.{R}")
    else:
        print(f"\n  {DIM}Sem stubs em cache.{R}")

    # Modelos
    models = glob.glob("models/*.pt")
    print(f"\n  {B}Modelos:{R}")
    if models:
        for m in models:
            size = os.path.getsize(m) / (1024**2)
            print(f"    {GR}✓{R} {os.path.basename(m)}  {DIM}({size:.0f} MB){R}")
    else:
        print(f"    {RE}✗ Nenhum modelo encontrado.{R}")
        print(f"    {DIM}Corre: bash models/download_models.sh{R}")

    # Output videos
    outputs = list_videos("output_videos")
    print(f"\n  {B}Vídeos de saída ({len(outputs)}):{R}")
    for o in outputs[-5:]:  # últimos 5
        size = os.path.getsize(o) / (1024**2)
        print(f"    {BL}{os.path.basename(o)}{R}  {DIM}({size:.0f} MB){R}")

    input("\nEnter para continuar...")


# ── menu principal ─────────────────────────────────────────────────────────────

def main():
    while True:
        clear()
        header()
        print(f"  {calibration_status()}\n")
        print(f"  {CY}[1]{R}  Processar jogo")
        print(f"  {CY}[2]{R}  Calibrar campo")
        print(f"  {CY}[3]{R}  Estado do sistema / stubs")
        print(f"  {CY}[0]{R}  Sair\n")

        raw = input(f"{B}Opção:{R} ").strip()

        if raw == "1":
            screen_process()
        elif raw == "2":
            screen_calibrate()
        elif raw == "3":
            screen_status()
        elif raw == "0":
            print(f"\n{DIM}Até logo.{R}\n")
            sys.exit(0)
        else:
            print(f"{RE}Opção inválida.{R}")


if __name__ == "__main__":
    main()
