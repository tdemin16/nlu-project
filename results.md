# Results
## GRU
### Subjectivity - No Att
| Fold | Acc | F1 |
|---|---|---|
|0|0,908|0,910|
|**1**|0,914|0,915|
|2|0,908|0,908|
|3|0,897|0,898|
|4|0,901|0,898|
|**Mean**|**90.6**|**90.6**|
|**Std**|**00.7**|**00.7**|

### Subjectivity - Att
| Fold | Acc | F1 |
|---|---|---|
|0|0,891|0,890|
|**1**|0,911|0,914|
|2|0,909|0,909|
|3|0,903|0,905|
|4|0,911|0,913|
|**Mean**|**90.5**|**90.6**|
|**Std**|**00.8**|**01.0**|

### Polarity - No Att No Filter
| Fold | Acc | F1 |
|---|---|---|
|0|0,815|0,814|
|1|0,815|0,808|
|2|0,828|0,827|
|3|0,812|0,812|
|4|0,830|0,841|
|**Mean**|**82.0**|**82.0**|
|**Std**|**00.8**|**01.4**|

### Polarity - Att No Filter
| Fold | Acc | F1 |
|---|---|---|
|0|0,848|0,852|
|1|0,828|0,817|
|2|0,848|0,857|
|3|0,828|0,841|
|4|0,853|0,849|
|**Mean**|**84.1**|**84.3**|
|**Std**|**01.2**|**01.6**|

### Polarity - No Att Filter
| Fold | Acc | F1 |
|---|---|---|
|0|0,772|0,762|
|1|0,770|0,770|
|2|0,750|0,722|
|3|0,760|0,747|
|4|0,820|0,810|
|**Mean**|**77.4**|**76.2**|
|**Std**|**02.7**|**03.2**|

### Polarity - Att Filter
| Fold | Acc | F1 |
|---|---|---|
|0|0,802|0,802|
|1|0,780|0,796|
|2|0,770|0,749|
|3|0,812|0,815|
|4|0,830|0,827|
|**Mean**|**79.9**|**79.8**|
|**Std**|**02.4**|**03.0**|


## Transformer
### Subjectivity
| Fold | Acc | F1 |
|---|---|---|
|0|0,963|0,963|
|**1**|0,974|0,974|
|2|0,970|0,970|
|3|0,967|0,966|
|4|0,973|0,973|
|**Mean**|**96.9**|**96.9**|
|**Std**|**00.5**|**00.5**|

### Polarity - No Filter
| Fold | Acc | F1 |
|---|---|---|
|0|0,915|0,909|
|1|0,870|0,869|
|2|0,880|0,882|
|3|0,897|0,899|
|4|0,850|0,847|
|**Mean**|**88.2**|**88.1**|
|**Std**|**02.5**|**02.5**|

### Polarity - Filter
| Fold | Acc | F1 |
|---|---|---|
|0|94.3|94.2|
|1|91.2|91.1|
|2|92.7|92.7|
|3|93.5|93.4|
|4|96.3|96.3|
|**Mean**|**93.6**|**93.5**|
|**Std**|**1.9**|**1.9**|


## Custom sentences
- My favourite film is 'Wir Kinder von Bahnhof Zoo'. The film is set in Berlin in the 1970s, where the scourge of child prostitution and juvenile drug addiction is told and made known in the western world. It is a film based on a true story, where the main character Christian played by Natja Brunckhorst is the only professional actress, in fact most of the other actors were taken from the streets, so as to make the scenes of the film even more real and crude. This aspect is what makes me love this unfiltered film, where the dark and deeply degraded side of a modern city like Berlin is shown for the first time in the history of cinema.
- Murder on the Orient Express is a 2017 film directed by Kenneth Branagh based on the book of the same name by Agatha Christie. A man is found dead in a carriage of a luxury train travelling through Europe. The journey is then interrupted to find out who the culprit is and Hercule Poirot is the man assigned to solve the mystery. I particularly liked this film because it adhered to the original plot of the book and thus succeeded in leaving the viewer in suspense for the duration of the film. The actors got into the part well and managed to create suspicions in me that were later disproved.
- Parasite is a 2019 Korean film directed by Bong Joon-ho. In the film, a poor family manages to get hired through deception by a rich family by posing as a university student, artist, driver and housekeeper. One day the wealthy family goes on holiday and the other family takes the opportunity to enjoy the luxury of the house, but the party is soon interrupted. Despite winning many awards, the film was unfortunately not to my liking. I felt it lacked content and lacked credibility, so much so that I couldn't wait for it to end. The clash between rich and poor people was developed in a predictable and superficial manner, leaving no message at the end of the film.
- Interstellar is maybe the best movie I have ever watched. The movie is set in the future where diseases are destroying all kinds of harvest. In her bedroom, young Murphy discovers a strange phenomenon related to gravity which will later guide his father to new planets with the hope of finding a new home for humanity. The movie is written by the Nolan brothers, starring Matthew McConaughey, and with Music by Hans Zimmer. Putting these four people together is the recipe for an oscar film. They were even capable of depicturing the best black hole ever seen on the big screen, and the best part of it is that at the time, no photo of a black hole was available. To me, it is a masterpiece and everyone should watch it.
- Don't Look Up is a 2021 film directed by Adam McKay. The movie starts with two astronomers (Leonardo Di Caprio and Jennifer Lawrence) who discover a new meteorite that is directed to earth and that is big enough to extinguish a species. However, the president of the United States (who resembles Donald Trump, although female) together with a Billionaire CEO (which is a mixture of Elon Musk and Steve Jobs) will try to diminish the problem to get votes and money. The cast is composed of Leonardo Di Caprio, Jennifer Lawrence, Timothee Chalamet, Cate Blanchett, Meryl Streep, and Ariana Grande. It seems like they focused on finding the best available actors (and most expensive ones) in order to cover an undeveloped idea. The movie is so unrealistic that I had to force myself in order to finish it, and I cannot understand how it is possible that everyone found it a great movie. The director wants to explain how, nowadays, we don't care enough about natural phenomena that could lead to extinction. However, he does it too forcely making it a ridiculous film.

### Subjectivity detection
All objective sentences have been successfully filtered with except to "The director wants to explain how, nowadays, we don't care enough about natural phenomena that could lead to extinction."

### Polarity classification
The final model has been able to correctly predict all reviews with and without filtering objective sentences.

Output:
- positive
- positive
- negative
- positive
- negative