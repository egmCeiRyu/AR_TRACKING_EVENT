# Motion Pal — menu dinâmico de personagens AR

## Executar

Na pasta do projeto, execute `python -m http.server 8000` e abra http://localhost:8000.
A câmera requer localhost ou HTTPS. O carregamento dos modelos requer internet.

## Usar

Escolha um personagem no menu. Permita a câmera. Use キャラクター変更 para voltar e escolher outro.
Todos os personagens acompanham olhos, boca e braços. O botão 追跡ポイント mostra os pontos de rastreamento.

## Atualizar e ampliar o catálogo

A interface pública apresenta apenas a seleção dos personagens e a experiência AR.
Para adicionar personagens ou mudar nomes, cores e ordem, edite characters.json no repositório.
A ordem dos itens no arquivo define a ordem dos cartões. Publique o arquivo atualizado
junto com o site. O catálogo é recarregado ao abrir a página, voltar da câmera e retornar à janela.
Nenhuma configuração local do antigo editor é usada.

## Configuração

characters.json contém version: 1 e uma lista characters de 1 a 50 itens.
Cada item tem id único (letras, números, hífen ou sublinhado), name (até 30 caracteres),
description (até 70 caracteres), template e colors opcional.
Templates: gentleman, robot, cat, bunny, alien, bear.
Cores opcionais: skin, body, arm, accent, iris, bg, em formato #RRGGBB.
As miniaturas usam o mesmo desenho Canvas do AR. Novas artes com formas diferentes
precisam de um novo template de desenho; o catálogo permite criar variações das seis formas existentes.
