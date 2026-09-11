# Motion Pal — menu dinâmico de personagens AR

## Executar

Na pasta do projeto, execute `python -m http.server 8000` e abra http://localhost:8000.
A câmera requer localhost ou HTTPS. O carregamento dos modelos requer internet.

## Usar

Escolha um personagem no menu. Permita a câmera. Use キャラクター変更 para voltar e escolher outro.
Todos os personagens acompanham olhos, boca e braços. O botão 追跡ポイント mostra os pontos de rastreamento.

## Editar o menu

Clique em メニューを編集. O painel permite adicionar personagens, editar nomes e descrições,
escolher uma das seis formas, mudar cores, reordenar e excluir. この端末に保存 salva neste navegador.
設定を書き出す exporta characters.json. 設定を読み込む importa uma configuração.
公開設定に戻す remove a personalização local e recarrega o arquivo publicado.

As alterações locais NÃO modificam os arquivos no servidor nem o menu de outros visitantes.
Para publicar para todos, substitua characters.json no repositório pelo arquivo exportado e publique
as alterações no GitHub Pages ou na hospedagem usada. O menu recarrega esse JSON ao abrir,
ao voltar da câmera, ao retornar à janela e pelo botão 最新の設定を読み込む.
Um navegador com personalização local precisa usar 公開設定に戻す para ver a versão publicada.
O editor é local, sem senha ou acesso de gravação ao servidor. Um painel compartilhado online
precisaria de autenticação e armazenamento de servidor.

## Configuração

characters.json contém version: 1 e uma lista characters de 1 a 50 itens.
Cada item tem id único (letras, números, hífen ou sublinhado), name (até 30 caracteres),
description (até 70 caracteres), template e colors opcional.
Templates: gentleman, robot, cat, bunny, alien, bear.
Cores opcionais: skin, body, arm, accent, iris, bg, em formato #RRGGBB.
As miniaturas usam o mesmo desenho Canvas do AR. Novas artes com formas diferentes
precisam de um novo template de desenho; o editor cria variações das seis formas existentes.
