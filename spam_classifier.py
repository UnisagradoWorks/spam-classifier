import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dados de exemplo
corpus = [
    ("Oferta especial! Ganhe 50% de desconto hoje!", 1),  # spam
    ("Reunião marcada para amanhã às 10h", 0),  # não spam
    ("Ganhou um prêmio de loteria. Responda para reivindicar.", 1),  # spam
    ("Lembrete: Consulta médica amanhã às 14h.", 0),  # não spam
    ("Ganhe dinheiro rápido e fácil! Acesse agora.", 1),  # spam
    ("Emagreça 10kg em uma semana com nosso novo produto!", 1),  # spam
    ("Seu CPF foi sorteado! Parabéns, você ganhou um prêmio.", 1),  # spam
    ("Aumente seu score de crédito agora mesmo!", 1),  # spam
    ("Oferta imperdível: compre um iPhone por metade do preço.", 1),  # spam
    ("Receba milhares de seguidores reais nas redes sociais!", 1),  # spam
    ("Promoção exclusiva: viagem dos sonhos com desconto!", 1),  # spam
    ("Você foi selecionado para testar um produto inovador gratuitamente.", 1),  # spam
    ("Descubra o segredo para enriquecer rapidamente.", 1),  # spam
    ("Aviso urgente: sua conta bancária precisa ser verificada.", 1),  # spam
    ("Seu nome foi sorteado! Você ganhou um carro zero!", 1),  # spam
    ("Limpe seu histórico de crédito em 24 horas!", 1),  # spam
    ("Ganhe um iPhone grátis apenas preenchendo este formulário.", 1),  # spam
    ("Promoção relâmpago: descontos de até 70% em eletrônicos.", 1),  # spam
    ("Invista em criptomoedas e obtenha retornos surpreendentes.", 1),  # spam
    ("Conquiste a casa própria sem burocracias. Saiba como!", 1),  # spam
    ("Você acaba de ser premiado como o cliente do mês. Resgate seu prêmio!", 1),  # spam
    ("Ganhe uma viagem para dois para um destino paradisíaco.", 1),  # spam
    ("Aviso importante: sua assinatura expirou. Renove agora!", 1),  # spam
    ("Receba agora mesmo um cartão de crédito com limite alto.", 1),  # spam
    ("Promoção única: produtos de beleza com 50% de desconto.", 1),  # spam
    ("Participe do nosso programa de afiliados e ganhe comissões incríveis.", 1),  # spam
    ("Empréstimo pessoal aprovado! Dinheiro na sua conta em 24 horas.", 1),  # spam
    ("Seja o primeiro a conhecer nossa oferta exclusiva. Não perca!", 1),  # spam
    ("Acesse este link para ganhar um voucher de compras.", 1),  # spam
    ("Descubra como ganhar dinheiro trabalhando em casa.", 1),  # spam
    ("Sorteio especial: ganhe um smartphone de última geração.", 1),  # spam
    ("Urgente: sua presença é requerida em nossa promoção especial.", 1),  # spam
    ("Oferta limitada: descontos incríveis em roupas de marca.", 1),  # spam
    ("Esta é uma mensagem regular de notificação.", 0),  # não spam
    ("Sua fatura mensal está disponível para visualização.", 0),  # não spam
    ("Confirme sua inscrição clicando no link abaixo.", 0),  # não spam
    ("Atualização de software disponível para download.", 0),  # não spam
    ("A reunião foi remarcada para outra data.", 0),  # não spam
    ("Você recebeu uma nova mensagem no seu e-mail.", 0),  # não spam
    ("A entrega do seu pedido está prevista para amanhã.", 0),  # não spam
    ("Atualização de segurança necessária para sua conta.", 0),  # não spam
    ("Confira as últimas notícias da semana em nosso boletim informativo.", 0),  # não spam
    ("Este é um lembrete para o evento de amanhã.", 0),  # não spam
    ("Informamos que seu voo está confirmado.", 0),  # não spam
]

# Separar os dados em features (X) e rótulos (y)
X, y = zip(*corpus)

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vetorizar as mensagens de texto usando a abordagem Bag of Words
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Criar e treinar o classificador Naive Bayes
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Fazer previsões no conjunto de teste
y_pred = classifier.predict(X_test_vectorized)

# Avaliar o desempenho do classificador
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Acurácia: {accuracy}')
print(f'Matriz de Confusão:\n{conf_matrix}')
print(f'Relatório de Classificação:\n{classification_rep}')