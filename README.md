# ONJ Seminar 2

## Predpriprava

- Prenesemo [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)
- Zaženemo CoreNLP v obliki strežnika z ukazom
```
java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 60000
```
- Kloniramo ta repozitorij

## Uporaba

### Način 1

- Zaženemo server.py v mapi cross validation
- Počakamo, da se modeli natrenirajo
- Na naslov [http://localhost:8080/predict](http://localhost:8080/predict) pošljemo zahtevo oblike:
```
{
"modelId": "A",
"question": "How does Shiranna feel as the shuttle is taking off?",
"questionResponse": "Shiranna feels both excited and nervous as the shuttle is taking off."
}

```
- Strežnik vrne rezultat (score), ki je lahko 0, 0.5 ali 1
- Na voljo so modeli A, B in C, iz vsake kategorije je uporabljena najboljša verzija (izbrani so najboljši parametri)


### Način 2

- V mapi corss validation se nahajajo različni modeli, ki jih lahko samostojno poganjamo
- Na vrhu vsake datoteke lahko spreminjamo parametre
- Treba je tudi klicati ustrezne funkcije na dnu datoteke

## Opombe

- Modeli, ki uporabljajo CoreNLP, rabijo nekaj časa da se natrenirajo
