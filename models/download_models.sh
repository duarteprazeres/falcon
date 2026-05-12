#!/bin/bash
# Download dos modelos especializados da Roboflow para análise de futebol
# Requer: pip install gdown

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "A descarregar modelos para $DIR ..."

# Modelo de deteção de jogadores (players, goalkeepers, referees, ball)
gdown -O "$DIR/football-player-detection.pt" "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"

# Modelo de deteção do campo (32 keypoints — o mais crítico para perspetiva dinâmica)
gdown -O "$DIR/football-pitch-detection.pt" "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"

# Modelo de deteção da bola (otimizado para bolas pequenas em HD)
gdown -O "$DIR/football-ball-detection.pt" "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"

echo ""
echo "✅ Modelos descarregados:"
ls -lh "$DIR"/*.pt 2>/dev/null || echo "⚠️  Nenhum .pt encontrado — verifica se o gdown está instalado."
