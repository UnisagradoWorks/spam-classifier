
# 💻 Classificador de Spam usando Árvore de Decisão
- Este é um projeto simples de classificação de mensagens em spam ou não spam utilizando a biblioteca scikit-learn em Python. O classificador é baseado em Árvore de Decisão e utiliza a representação de texto em vetores com o CountVectorizer.

# 🧩 Configuração do Ambiente
Certifique-se de ter o Python instalado em sua máquina. Você pode instalar as bibliotecas necessárias usando o seguinte comando:

```bash
pip install scikit-learn
```
# 🕹 Como Usar
1. Clone o repositório: 
```bash
git clone https://github.com/UnisagradoWorks/spam-classifier
```  
2. Dados de exemplo:
```bash
nova_mensagem = [
    ("Oferta especial! Ganhe 50% de desconto hoje!", 1),  # 1 para spam
    ("Reunião marcada para amanhã às 10h", 0),  # 0 para não spam
    ("Ganhou um prêmio de loteria. Responda para reivindicar.", 1),
    ("Lembrete: Consulta médica amanhã às 14h.", 0),
    # Adicione mais títulos de exemplo conforme necessário
]
```
3. Execute o código:
```bash
spam_classifier.py
```
# 📊 Resultados
Após a execução do código, os resultados, incluindo a acurácia e o relatório de classificação, serão exibidos no console.
