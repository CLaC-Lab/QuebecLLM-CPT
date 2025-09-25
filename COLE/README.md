---
license: mit
language:
- fr
- en
configs:
- config_name: allocine
  data_files:
  - split: train
    path: allocine/*train*.jsonl
  - split: validation
    path: allocine/*validation*.jsonl
  - split: test
    path: allocine/*test*.jsonl
- config_name: paws_x
  data_files:
  - split: train
    path: paws_x/*train*.jsonl
  - split: validation
    path: paws_x/*validation*.jsonl
  - split: test
    path: paws_x/*test*.jsonl
- config_name: fquad
  data_files:
  - split: validation
    path: fquad/*validation*.jsonl
  - split: test
    path: fquad/*test*.jsonl
- config_name: gqnli
  data_files:
  - split: train
    path: gqnli/*train*.jsonl
  - split: validation
    path: gqnli/*validation*.jsonl
  - split: test
    path: gqnli/*test*.jsonl
- config_name: lingnli
  data_files:
  - split: train
    path: lingnli/lingnli_train.jsonl
  - split: test
    path: lingnli/lingnli_test.jsonl
- config_name: multiblimp
  data_files:
  - split: train
    path: multiblimp/*train*.jsonl
  - split: validation
    path: multiblimp/*validation*.jsonl
  - split: test
    path: multiblimp/*test*.jsonl
- config_name: piaf
  data_files:
  - split: train
    path: piaf/*train*.jsonl
  - split: validation
    path: piaf/*validation*.jsonl
  - split: test
    path: piaf/*test*.jsonl
  features:
  - name: id
    dtype: string
  - name: title
    dtype: string
  - name: context
    dtype: string
  - name: question
    dtype: string
  - name: answers
    sequence:
    - name: text
      dtype: string
    - name: answer_start
      dtype: int32
- config_name: sickfr
  data_files:
  - split: train
    path: sickfr/*train*.jsonl
  - split: validation
    path: sickfr/*validation*.jsonl
  - split: test
    path: sickfr/*test*.jsonl
- config_name: xnli
  data_files:
  - split: train
    path: xnli/*train*.jsonl
  - split: validation
    path: xnli/*validation*.jsonl
  - split: test
    path: xnli/*test*.jsonl
- config_name: qfrcola
  data_files:
  - split: train
    path: qfrcola/*train*.jsonl
  - split: validation
    path: qfrcola/*dev*.jsonl
  - split: test
    path: qfrcola/*test*.jsonl
- config_name: qfrblimp
  data_files:
  - split: train
    path: qfrblimp/*train*.jsonl
  - split: validation
    path: qfrblimp/*validation*.jsonl
  - split: test
    path: qfrblimp/*test*.jsonl
- config_name: sts22
  data_files:
  - split: train
    path: sts22/*train*.jsonl
  - split: test
    path: sts22/*test*.jsonl
- config_name: qfrcore
  data_files:
  - split: test
    path: qfrcore/*test*.jsonl
- config_name: qfrcort
  data_files:
  - split: test
    path: qfrcort/qfrcort.jsonl
- config_name: daccord
  data_files:
  - split: test
    path: daccord/daccord_test.jsonl
- config_name: mnli-nineeleven-fr-mt
  data_files:
  - split: test
    path: mnli-nineeleven-fr-mt/multinli_nineeleven_fr_mt_test.jsonl
- config_name: french_boolq
  data_files:
  - split: test
    path: french_boolq/french_boolq_test.jsonl
- config_name: rte3-french
  data_files:
    - split: validation
      path: rte3-french/*dev*.jsonl
    - split: test
      path: rte3-french/*test*.jsonl
- config_name: fracas
  data_files:
   - split: test
     path: fracas/fracas_test.jsonl
- config_name: wino_x_lm
  data_files:
   - split: test
     path: wino_x_lm/wino_x_lm_test.jsonl
- config_name: wino_x_mt
  data_files:
   - split: test
     path: wino_x_mt/wino_x_mt_test.jsonl
- config_name: mms
  data_files:
   - split: test
     path: mms/mms_test.jsonl
- config_name: wsd
  data_files:
  - split: train
    path: wsd/*train*.jsonl
  - split: test
    path: wsd/*test*.jsonl
pretty_name: COLE
---
---
title: COLE !
emoji: 🐳
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
---

## COLE Dataset Card
## Dataset Summary

The COLE benchmark is a suit of multiple French NLP tasks for evaluating language models. It includes test sets, and some validation, and training sets for tasks such as sentiment analysis, question answering, NLI, and more.

## Task Descriptions

## Allocine.fr
Allo-ciné tests language understanding in sentiment classification by feeding movie reviews which can be either positive and negative, the task consists in giving the correct sentiment for each review.

## DACCORD
Determine if a French sentence makes sense semantically (binary label).

## FQuAD
Fquad is question/answer pair built on high-quality wikipedia articles. The goal of the model in this task is to accurately predict if the answer to the question really can be found in the provided answer.

## FraCaS 
Fracas is a natural language inference (NLI) taskthe where the model must classify the relationship between a premise and a hypothesis-entailment, contradiction, or neutral-based on complex linguistic phenomena such as quantifiers, plurality, anaphora, and ellipsis.

## Fr-BoolQ
Boolean question answering in French: answer true/false based on context.

## GQNLI-fr
The dataset consists of carefully constructed premise-hypothesis pairs that involve quantifier logic (e.g. most, at least, more than half). The goal is to evaluate the model's ability to reason about these expressions and determine whether the hypothesis logically follows from the premise, contradicts it, or is neutral.

## LingNLI
LingNLI is a NLI corpus collected by putting a linguist 'in the loop' to dynamically introduce novel constraints during data collection, aiming to mitigate the systematic gaps and biases often found in crowdsourced datasets.

## MMS-fr 
MMS-fr is a sentiment analysis task where the model classifies a French text as positive (2), neutral (1), or negative (0), assessing its ability to detect sentiment across diverse domains and sources.

## MNLI-nineeleven-Fr-MT 
French machine-translated version of MNLI using 9/11 context, for entailment classification.

## MultiBLiMP-Fr 
MultiBLiMP-Fr is a grammatical judgment task where the model must identify the grammatically correct sentence from a minimal pair differing by a single targeted feature, thereby assessing its knowledge of French syntax, morphology, and agreement.

## PAWS-X
This task aims to test paraphrase identification by giving two sentences and a label defining if these sentences are equivalent in meaning or not.

## PIAF
This task consists of pairs of questions and text answers with information of where in the answer is the truly relevant information.

## QFrBLiMP
This task gives the model sentences pairs, the goal is to determine if the sentences are semantically equivalent, or, put more simply, if they mean the same thing, even with slightly different syntax and words.

## QFrCoLA 
QFrCoLA is a french dataset made from multiple french language sites such as académie-française.fr and vitrinelinguistique.com. It aims to tests models ability to determine a sentence's acceptability in french on subjects such as grammar and syntax. The answer is a binary label indicating if the sentence is correct or not.

## QFrCoRE
QFrCoRE is a definition matching task where the model selects the correct standard French  definition for a Quebec French expression from a list of candidates.

## QFrCoRT
QFrCoRE is a definition matching task where the model selects the correct standard French  definition for a Quebec French term from a list of candidates.

## RTE3-Fr
French version of RTE3 for textual entailment recognition.

## SICK-fr
This task also has pairs of sentences and notes them on 2 dimensions, relatedness and entailment. While relatedness scales from 1 to 5, entailement is a choice between entails, contradicts or neutral.

## STS22
This task evaluates whether pairs of news articles, written in different languages, cover the same story. It focuses on document-level similarity, where systems rate article pairs on a 4-point scale from most to least similar

## Wino-X-LM
Pronoun resolution task: choose between two referents in a sentence with an ambiguous pronoun.

## Wino-X-MT
Translation-based pronoun resolution: choose which of two French translations uses the correct gendered pronoun.

## WSD-Fr
WSD-Fr is a word sense disambiguation task where the model must identify the correct meaning of an ambiguous verb in context, as part of the FLUE benchmark.

## XNLI-fr
This task consists of pairs of sentences where the goal is to determine the relation between the two sentences, this relation can be either entailement, neutral or contradiction.


## Language
The language data in COLE is in French .

### Dataset structure

## Allocine.fr
```json

{
  "review": "Magnifique épopée, une belle histoire, touchante avec des acteurs qui interprètent très bien leur rôles (Mel Gibson, Heath Ledger, Jason Isaacs...), le genre de film qui se savoure en famille! :)",
  "label": 1
}
```

## DACCORD
```json

{
  "id": "a001",
  "premise": "Le camion-remorque de la vidéo transporte un long tube cylindrique, qui est une pièce destinée à une raffinerie de pétrole en Ouzbékistan.",
  "hypothesis": "Le camion-remorque de la vidéo transporte un missile nucléaire russe.",
  "label": "1",
  "label_text": "contradiction",
  "url": "https://factuel.afp.com/doc.afp.com.32MJ3M7-1",
  "genre": "conflit ukrainien-russe"
}
```

## FQuAD
```json

{
  "title": "pégase_23_3",
  "context": "D'anciennes théories associent Pégase au combat naval, ou voient en lui un simple navire...",
  "question": "Quand le théologien Jacques-Paul Migne s'exprime au sujet de Méduse ?",
  "answers": {
    "answers_start": [509, 512, 512],
    "text": ["en 1855", "1855", "1855"]
  },
  "is_impossible": false
}
```

## FraCaS
```json

{
  "id": "1",
  "premise": "Un Italien est devenu le plus grand ténor du monde.",
  "hypothesis": "Il y a eu un Italien qui est devenu le plus grand ténor du monde.",
  "label": "0",
  "question": "Y a-t-il eu un Italien qui soit devenu le plus grand ténor du monde ?",
  "answer": "yes",
  "premises_original": "An Italian became the world's greatest tenor.",
  "premise1": "Un Italien est devenu le plus grand ténor du monde.",
  "premise1_original": "An Italian became the world's greatest tenor.",
  "premise2": "",
  "premise2_original": "",
  "premise3": "",
  "premise3_original": "",
  "premise4": "",
  "premise4_original": "",
  "premise5": "",
  "premise5_original": "",
  "hypothesis_original": "There was an Italian who became the world's greatest tenor.",
  "question_original": "Was there an Italian who became the world's greatest tenor?",
  "note": "",
  "topic": "GENERALIZED QUANTIFIERS"
}
```

## Fr-BoolQ
```json

{
  "question": "Jobs avait-il des sautes d'humeur inimaginables durant la période où il dirigeait NeXT ?",
  "passage": "Il a beaucoup été question de la personnalité agressive et exigeante de Steve Jobs. [...] Dan’l Lewin, déclare dans ce même magazine que Steve Jobs, durant cette période, « avait des sautes d'humeur inimaginables » [...]",
  "label": 1
}
```

## GQNLI-Fr
```json

{
  "uid": 214,
  "premise": "Il y a six ours. Trois ours marron, un ours noir et un ours blanc courent le long de l'herbe cyan.",
  "hypothesis": "Un ours beige court.",
  "label": 1,
  "label_text": "neutral",
  "premise_original": "There are six bears...",
  "hypothesis_original": "One beige bear runs."
}
```

## LingNLI
```json

{
  "premise": "La richesse des citations verbatim - constituant un bon tiers de ce livre - améliore également la vraisemblance de Burn Rate.",
  "hypothesis": "Burn Rate manque de véracité et n'inclut aucune référence à d'autres œuvres d'aucune sorte.",
  "label": 2
}
```

## MMS
```json

{
  "text": "Cadeaux pour ma fille.",
  "label": 2
}
```

## MNLI-nineeleven-Fr-MT
```json

{
  "premise": "La faillite du nationalisme laïque et autocratique était évidente dans le monde musulman à la fin des années 1970.",
  "hypothesis": "Les musulmans détestaient le nationalisme autocratique à la fin des années 1970.",
  "label": "1",
  "label_text": "neutral",
  "pairID": "62534e",
  "promptID": "62534",
  "premise_original": "The bankruptcy of secular, autocratic nationalism was evident across the Muslim world by the late 1970s.",
  "hypothesis_original": "Muslims disliked autocratic nationalism by the late 1970s."
}
```

## MultiBLiMP-Fr
```json

{
  "sentence_a": "C'est le genre à lequel appartiennent les espèces de kiwi.",
  "sentence_b": "C'est le genre à lequel appartenez les espèces de kiwi.",
  "label": 0
}
```

## PAWS-X
```json

{
  "id": 12,
  "sentence1": "La rivière Tabaci est un affluent de la rivière Leurda en Roumanie.",
  "sentence2": "La rivière Leurda est un affluent de la rivière Tabaci en Roumanie.",
  "label": 0
}
```

## PIAF
```json

{
  "id": "p140295203922856",
  "title": "Alaungpaya",
  "context": "Il ne convainquit cependant pas tout le monde. Après la chute d'Ava le 23 mars 1752, son propre père lui conseilla de se soumettre : il lui fit valoir que, bien qu'ayant des quantités de soldats enthousiastes, il manquait de mousquets et que leur petite palissade ne résisterait jamais à une armée bien équipée qui venait de mettre à sac Ava, puissamment fortifiée. Alaungpaya, impavide, déclara : « Quand on combat pour son pays, il importe peu qu'on soit rares ou nombreux. ce qui compte est que vos camarades aient un cœur sincère et des bras forts. » Il prépara sa défense en fortifiant Moksobo (renommé Shwebo), avec une palissade et des douves. Il fit couper la forêt à l'extérieur, détruire les mares et combler les puits.",
  "question": "De quoi Alaungpaya aurait il eu besoin pour remporter la bataille ?",
  "answers": {
    "text": ["de mousquets et que leur petite palissade"],
    "answer_start": [222]
  }
}
```

## QFrBLiMP
```json

{
  "id": 250,
  "label": 0,
  "ungrammatical": "Cette femme chante très haute.",
  "source": "https://vitrinelinguistique.oqlf.gouv.qc.ca/...",
  "category": "morphology",
  "type": 11,
  "subcat": 13.0,
  "grammatical": "Cette femme chante très haut.",
  "options": [
    {"id": "1", "text": "La phrase numéro 1"},
    {"id": "2", "text": "La phrase numéro 2"}
  ],
  "answer": "accept"
}
```

## QFrCoLA
```json

{
  "label": 1,
  "sentence": "Je vous en prie, soyez bref.",
  "source": "https://vitrinelinguistique.oqlf.gouv.qc.ca/...",
  "category": "anglicism"
}
```

## QFrCoRE
```json

{
  "expression": "Avoir la chienne",
  "choices": [
    "Prendre une chaise et s'asseoir.",
    "Avoir du plaisir, parfois avec une connotation sexuelle.",
    "Prépare-toi, ça va brasser.",
    "Tomber amoureux.",
    "Être en pleine forme.",
    "Critiquer sévèrement.",
    "Personne inefficace, qui ne travaille pas bien.",
    "Il se comporte mal en public.",
    "Se détendre, arrêter de s'énerver.",
    "Avoir peur."
  ],
  "correct_index": 9,
  "reference": "https://canada-media.ca/expressions-quebecoises/"
}
```

## QFrCoRT
```json
{
  "terme": "Adonner",
  "choices": [
    "se payer du bon temps",
    "tu sais",
    "Voici quelques éléments typiques pour décrire l'hiver québécois :La bordée de neige(tempête de neige) de la fin décembre nous a laissédes bancs de neige(congères) sur le bord des rues. Nous avons eu quelques épisodes depoudrerie(blizzard) qui ont rendu les déplacements difficiles, surtout en voiture. Mais c'est vraimentla glace noire(verglas) qui cause le plus d'accidents. Il faudra attendre jusqu'auredoux(remontée des températures) pour que la neige et la glace se transforment ensloche(gadoue constituée de neige fondante et d'eau) puis disparaissent au retour du printemps.",
    "En hiver, il ne faut pas s'encabaner!Ce joli verbe vient du nom \" cabane \" qui désigne un petit espace de rangement. S'encabaner, c'est donc \" rester dans sa cabane (sa maison), ne pas sortir, rester cloitré chez soi \". Mais comme le disaient les membres du groupe Mes Aïeux dans leur chanson \" Dégénération \" : \" Il ne faut pas rester encabané! \" (surtout en hiver).",
    "On ne parle pas ici de la neige de la veille.Cette expression signifie \" avoir de l'expérience,voir venir les choses \".",
    "Ça n'a étrangement absolument rien à voir avec le fait qu'il manque quelque chose. Ben manque se veut plutôt un synonyme de \" peut-être \" , \" sûrement \" ou \" probablement \" . Particulièrement utilisé du côté nord de la Gaspésie et sur la pointe, ben manque que tu risques de l'entendre si tu te promènes dans ces coins-là.",
    "gant de toilette",
    "Unpoisson d'avrilest une plaisanterie que l'on fait le 1er avril à une connaissance.",
    "avoir de la monnaie",
    "Le verbe \" adonner \" s'utilise pour parler de quelque chose qui se produit de façon fortuite, d'une coïncidence. Il peut avoir différentes nuances de sens selon le contexte.Exemples : \" Tu vas à Québec cette fin de semaine? Ça adonne que moi aussi! Faisons du covoiturage! \"\" Je devais commencer mes cours de zumba ce soir mais ça adonne mal : mon fils est malade! \""
  ],
  "correct_index": 9,
  "reference": "https://vivreenfrancais.mcgill.ca/capsules-linguistiques/expressions-quebecoises/"
}
```

## RTE3-Fr
```json

{
  "id": "1",
  "language": "fr",
  "premise": "La vente a été faite pour payer la facture fiscale de 27,5 milliards de dollars de Yukos, Yuganskneftegaz a été vendu à l'origine pour 9,4 milliards de dollars à une entreprise peu connue, Baikalfinansgroup, qui a ensuite été rachetée par la compagnie pétrolière publique russe Rosneft.",
  "hypothesis": "Baikalfinansgroup a été vendu à Rosneft.",
  "label": "0",
  "label_text": "entailment",
  "task": "IE",
  "length": "short"
}
```

## SICK-Fr
```json

{
  "Unnamed: 0": 5,
  "label": 2,
  "relatedness_score": 3.2999999523,
  "sentence_A": "Deux chiens se battent et se câlinent.",
  "sentence_B": "Il n'y a pas de lutte et d'étreinte de chiens."
}
```

## STS22
```json

{
  "id": "1559147599_1558534688",
  "score": 1.0,
  "sentence1": "KABYLIE (TAMURT) – Les répercussions économiques...",
  "sentence2": "Le décret n° 2020-293 du 23 mars 2020..."
}
```

## Wino-X-LM
```json

{
  "qID": "3UDTAB6HH8D37OQL3O6F3GXEEOF09Z-1",
  "sentence": "The woman looked for a different vase for the bouquet because it was too small.",
  "context_en": "The woman looked for a different vase for the bouquet because _ was too small.",
  "context_fr": "La femme a cherché un vase différent pour le bouquet car _ était trop petit.",
  "option1_en": "the bouquet",
  "option2_en": "the vase",
  "option1_fr": "le bouquet",
  "option2_fr": "le vase",
  "answer": 2,
  "context_referent_of_option1_fr": "bouquet",
  "context_referent_of_option2_fr": "vase"
}
```

## Wino-X-MT
```json

{
  "qID": "3FULMHZ7OUVKJ7S9R0LMS753751M44-1",
  "sentence": "As the wolf approached the house, the man quickly took the knife and not the gun to defend himself because it was near him.",
  "translation1": "Alors que le loup s'approchait de la maison, l'homme prit rapidement le couteau et non l'arme pour se défendre car il était près de lui.",
  "translation2": "Alors que le loup s'approchait de la maison, l'homme prit rapidement le couteau et non l'arme pour se défendre car elle était près de lui.",
  "answer": 1,
  "pronoun1": "il",
  "pronoun2": "elle",
  "referent1_en": "knife",
  "referent2_en": "gun",
  "true_translation_referent_of_pronoun1_fr": "couteau",
  "true_translation_referent_of_pronoun2_fr": "arme",
  "false_translation_referent_of_pronoun1_fr": "couteau",
  "false_translation_referent_of_pronoun2_fr": "arme"
}
```

## WSD-Fr
```json

{
  "sentence": "Il rend hommage au roi de France et des négociations aboutissent au traité du Goulet , formalisant la paix entre les deux pays .",
  "labels_idx": [10],
  "label": "négociations"
}
```

## XNLI-Fr
```json

{
  "premise": "Ils m'ont dit qu'à la fin, on m'amènerait un homme pour que je le rencontre.",
  "hypothesis": "Le gars arriva un peu en retard.",
  "label": 1
}
```

## Allocine.fr
| split      | # examples |
|------------|-----------:|
| train      |             |
| validation |       20,000 |
| test       |       20,000 |

## DACCORD
| split      | # examples |
|------------|-----------:|
| test       |       1,034 |

## FQuAD
| split      | # examples |
|------------|-----------:|
| validation |       100 |
| test       |       400 |

## FraCaS
| split      | # examples |
|------------|-----------:|
| test       |       346 |

## Fr-BoolQ
| split      | # examples |
|------------|-----------:|
| test       |        178 |

## GQNLI-Fr
| split      | # examples |
|------------|-----------:|
| train      |       243 |
| validation |        27 |
| test       |        30 |

## LingNLI
| split      | # examples |
|------------|-----------:|
| train      |     29,985 |
| test       |      4,893 |

## MMS
| split      | # examples   |
|------------|-------------:|
| train      |      132,696 |
| validation |       14,745 |
| test       |       63,190 |

## MNLI-nineeleven-Fr-MT
| split      | # examples |
|------------|-----------:|
| test       |       2,000 |

## MultiBLiMP-Fr
| split      | # examples |
|------------|-----------:|
| train      |        160 |
| validation |         18 |
| test       |         77 |

## PAWS-X
| split      | # examples |
|------------|-----------:|
| train      |     49,401  |
| validation |       2,000 |
| test       |       2,000 |

## PIAF
| split      | # examples |
|------------|-----------:|
| train      |       3,105 |
| validation |        346 |
| test       |        384 |

## QFrBLiMP
| split      | # examples |
|------------|-----------:|
| train      |       NA   |
| validation |       2,061 |
| test       |       2,290 |

## QFrCoLA
| split      | # examples |
|------------|-----------:|
| train      |      15,846 |
| validation |       1,761 |
| test       |       7,546 |

## QFrCoRE
| split      | # examples |
|------------|-----------:|
| test       |       4,633 |

## QFrCoRT
| split      | # examples |
|------------|-----------:|
| test       |        201 |

## rte3-Fr
| split      | # examples   |
|------------|-------------:|
| train      |      269,821 |
| validation |         800 |
| test       |        3,121 |

## SICK-fr
| split      | # examples |
|------------|-----------:|
| train      |       4,439 |
| validation |        495 |
| test       |       4,906 |

## STS22
| split      | # examples |
|------------|-----------:|
| train      |        101 |
| test       |         72 |

## Wino-X-LM
| split      | # examples |
|------------|-----------:|
| test       |       2,793 |

## Wino-X-MT
| split      | # examples |
|------------|-----------:|
| test       |       2,988 |

## WSD
| split      | # examples |
|------------|-----------:|
| test       |       3,121 |
| train       |       269,821  |

## XNLI-Fr
| split      | # examples   |
|------------|-------------:|
| train      |      393,000 |
| validation |        2,490 |
| test       |        5,010 |



## Citation

TO ADD


