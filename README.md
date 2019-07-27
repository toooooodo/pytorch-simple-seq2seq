# Pytorch Implementation of Simple Seq2Seq with Attention

A Seq2seq network to translate from French to English.

Pytorch version: 1.1.0

**Reference:**
- 《动手学深度学习》[10.12. 机器翻译](http://zh.d2l.ai/chapter_natural-language-processing/machine-translation.html)
- Pytorch Tutorials [TRANSLATION WITH A SEQUENCE TO SEQUENCE NETWORK AND ATTENTION](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

**Dataset:** 
- Tatoeba Project [French - English](http://www.manythings.org/anki/)

### **To evaluate, we sample 5 sentences randomly, and network performance are as follows:**

epoch | input sentence | target sentence | output sentence
:-:|:-:|:-:|:-:
30|nous sommes deja en retard .|we re already late .|we re all proud of us . <eos>
30|on s occupe bien d eux .|they are well looked after .|they re jealous of us . <eos>
30|tu agis comme un enfant .|you re acting like a child .|you re acting like a child . <eos>
30|elles sont toutes normales .|they re all normal .|they re all dead . <eos>
30|e ne te donnerai aucun argent .|i m not giving you any money .|i m not giving you any more money . <eos>
70|nous sommes deja en retard .|we re already late .|we re all late . <eos>
70|on s occupe bien d eux .|they are well looked after .|she s a bit in agreement . <eos>
70|tu agis comme un enfant .|you re acting like a child .|you re acting like a child . <eos>
70|elles sont toutes normales .|they re all normal .|they re all mad . <eos>
70|e ne te donnerai aucun argent .|i m not giving you any money .|i m not giving you any money . <eos>
100|nous sommes deja en retard .|we re already late .|we re already late . <eos>
100|on s occupe bien d eux .|they are well looked after .|they are well looked after . <eos>
100|tu agis comme un enfant .|you re acting like a child .|you re acting like a child . <eos>
100|elles sont toutes normales .|they re all normal .|they re all normal . <eos>
100|e ne te donnerai aucun argent .|i m not giving you any money .|i m not giving you any money . <eos>
