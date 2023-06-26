
def bot_orbital(tempo, dados_orbita):
    import pywhatkit
    import keyboard
    import time
    from datetime import datetime

    # contato
    contato = '+5547992344337'

    # mensagem
    mensagem = '|----------------| ' \
               '\n' \
               '| Nova simulacao | ' \
               '\n' \
               '|----------------| ' \
               '\n' \
               f'Tempo de inicio: {tempo[0]} ' \
               f'\n' \
               f'Tempo final: {tempo[1]} ' \
               f'\n' \
               f'Tempo total de simulacao: {tempo[2]} ' \
               f'\n' \
               f'|-----------------| ' \
               f'\n' \
               '| Dados da orbita: | ' \
               '\n' \
               '|------------------| ' \
               '\n' \
               f'Semi eixo = {dados_orbita[0]} ' \
               f'\n' \
               f'Excentricidade = {dados_orbita[1]} ' \
               f'\n' \
               f'Raan = {dados_orbita[2]} ' \
               f'\n' \
               f'Argumento do perigeu = {dados_orbita[3]} ' \
               f'\n' \
               f'Anomalia verdadeira = {dados_orbita[4]} ' \
               f'\n' \
               f'Inclinação = {dados_orbita[5]} ' \
               f'\n' \
               f'Numero de orbitas = {dados_orbita[6]} ' \
               f'\n' \
               f'Passo de integraçao = {dados_orbita[7]} ' \
               f'\n'
    # mensagem
    pywhatkit.sendwhatmsg(contato, mensagem, time_hour=datetime.now().hour, time_min=datetime.now().minute + 2)



    # figuras
    figuras = ['imagens_resultado/animacao_3d.png', 'imagens_resultado/radiacao_albedo.png',
               'imagens_resultado/radiacao_IR.png', 'imagens_resultado/radiacao_solar.png',
               'imagens_resultado/radiacao_total.png']
    # nomes
    nomes = ['animacao_3d', 'radiacao_albedo', 'radiacao_IR', 'radiacao_solar', 'radiacao_total']
    while len(figuras) >=1:
        # enviar mensagem
        # pywhatkit.sendwhatmsg(contato[0], "VAMOS AUTOMATIZAR TUDO!", datetime.now().hour,datetime.now().minute + 2)
        pywhatkit.sendwhats_image(contato, figuras[0], nomes[0], datetime.now().hour, datetime.now().minute, datetime.now().second + 5)
        del figuras[0]
        del nomes[0]
        time.sleep(10)
        keyboard.press_and_release('ctrl + w')

def bot_temperaturas(tempo):
    import pywhatkit
    import keyboard
    import time
    from datetime import datetime

    # contato
    contato = '+5547992344337'

    # mensagem
    mensagem = '|----------------| ' \
               '\n' \
               '| Nova simulacao | ' \
               '\n' \
               '|----------------| ' \
               '\n' \
               f'Tempo de inicio: {tempo[0]} ' \
               f'\n' \
               f'Tempo final: {tempo[1]} ' \
               f'\n' \
               f'Tempo total de simulacao: {tempo[2]} ' \
               f'\n' \
    # mensagem
    pywhatkit.sendwhatmsg(contato, mensagem, time_hour=datetime.now().hour, time_min=datetime.now().minute + 2)

    # figuras
    figuras = ['imagens_resultado/temp_faces.png']
    # nomes
    nomes = ['Temperatura nas Faces']
    while len(figuras) >= 1:
        # enviar mensagem
        # pywhatkit.sendwhatmsg(contato[0], "VAMOS AUTOMATIZAR TUDO!", datetime.now().hour,datetime.now().minute + 2)
        pywhatkit.sendwhats_image(contato, figuras[0], nomes[0], datetime.now().hour, datetime.now().minute,
                                  datetime.now().second + 5)
        del figuras[0]
        del nomes[0]
        time.sleep(10)
        keyboard.press_and_release('ctrl + w')