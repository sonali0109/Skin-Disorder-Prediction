# Skin-Disorder-Prediction
## Problem Statement

*   Create a predictive model using machine learning techniques to predict the various classes of skin disease. 

### **Dataset Information :**
This database contains 34 attributes, 33 of which are linear valued and one of them is nominal. The differential diagnosis of erythemato-squamous diseases is a real problem in dermatology. They all share the clinical features of erythema and scaling, with very little differences. The diseases in this group are psoriasis, seboreic dermatitis, lichen planus, pityriasis rosea, cronic dermatitis, and pityriasis rubra pilaris. Usually a biopsy is necessary for the diagnosis but unfortunately these diseases share many histopathological features as well. Another difficulty for the differential diagnosis is that a disease may show the features of another disease at the beginning stage and may have the characteristic features at the following stages. Patients were first evaluated clinically with 12 features. Afterwards, skin samples were taken for the evaluation of 22 histopathological features. The values of the histopathological features are determined by an analysis of the samples under a microscope.In the dataset constructed for this domain, the family history feature has the value 1 if any of these diseases has been observed in the family, and 0 otherwise. The age feature simply represents the age of the patient. Every other feature (clinical and histopathological) was given a degree in the range of 0 to 3. Here, 0 indicates that the feature was not present, 3 indicates the largest amount possible, and 1, 2 indicate the relative intermediate values.The names and id numbers of the patients were recently removed from the database.

### **Attribute Information :**
Clinical Attributes: (take values 0, 1, 2, 3, unless otherwise indicated)
*   1: erythema
*   2: scaling
*   3: definite borders
*   4: itching
*   5: koebner phenomenon
*   6: polygonal papules
*   7: follicular papules
*   8: oral mucosal involvement
*   9: knee and elbow involvement
*   10: scalp involvement
*   11: family history, (0 or 1)
*   Histopathological Attributes: (take values 0, 1, 2, 3)
*   12: melanin incontinence
*   13: eosinophils in the infiltrate
*   14: PNL infiltrate
*   15: fibrosis of the papillary dermis
*   16: exocytosis
*   17: acanthosis
*   18: hyperkeratosis
*   19: parakeratosis
*   20: clubbing of the rete ridges
*   21: elongation of the rete ridges
*   22: thinning of the suprapapillary epidermis
*   23: spongiform pustule
*   24: munro microabcess
*   25: focal hypergranulosis
*   26: disappearance of the granular layer
*   27: vacuolisation and damage of basal layer
*   28: spongiosis
*   29: saw-tooth appearance of retes
*   30: follicular horn plug
*   31: perifollicular parakeratosis
*   32: inflammatory monoluclear inflitrate
*   33: band-like infiltrate
*   34: Age (linear)

### **Suggestions to the Doctors to identify the skin diseases of the patient at the earliest :**

#### where feature is not present.
*   A Patient who has feature not present in itiching,koebner phenomnen,polygonal papules,folicular papules,oral mucosal involvement,melanin incontinece,eosinophils_in_the_infiltrate,PNL filtrate,fibrosis_of_the_papillary_dermis,exocythsis, hyperkeartosis,spongiform pustle,munro microbcess,focal hypergranulosis,disappearance_of_the_granular_layer, vacuolisation_and_damage_of_basal_layer,spongiosis,saw teeth apperance of the retes,folicular horn plug,perifolicular parakeratosis,band like infiltrate they all have highest chance of having class 1 (psoriasis) skin dieases problem.
*   A Patient who has feature not present in definite border,koebner phenomnen,polygonal papules,folicular papules,oral_mucosal_involvement,knee and elbow envolvement,scalp envolvement,melanin_incontinence, eosinophils_in_the_infiltrate,fibrosis_of_the_papillary_dermis,aconthosis,hyperkeratosis,perakeratosis, clubbing_of_the_rete_ridges,elongation_of_the_rete_ridges,thinning_of_the_suprapapillary_epidermis,spongiform_pustule, munro_microabcess,focal_hypergranulosis,disappearance_of_the_granular_layer,vacuolisation_and_damage_of_basal_layer,saw- tooth_appearance_of_retes,follicular_horn_plug,perifollicular_parakeratosis,inflammatory_monoluclear_inflitrate,band- like_infiltrate they have highest chance of having class 2 (seboreic dermatitis) skin dieases problem.
*   A Patient who has feature not present in koebner_phenomenon,folicular papules,knee and elbow envolvement,scalp envolvement,eosinophils_in_the_infiltrate,PNL inflitrate,fibrosis_of_the_papillary_dermis,hyperleratosis,perekeratosis, clubbing_of_the_rete_ridges,elongation_of_the_rete_ridges,thinning_of_the_suprapapillary_epidermis,spongiform pustule, munro_microabcess,disappearance_of_the_granular_layer,spongisis,perifollicular_parakeratosis they have highest chance of having class 3 (lichen planus) skin disorder problem.
*   A Patient who has feature not present in definite border,itiching,koebner_phenomenon,polygonals papules,folicular papules,oral_mucosal_involvement,knee_and_elbow_involvement,scalp involvement,melanin_incontinence, eosinophils_in_the_infiltrate,PNL infiltrate,fibrosis_of_the_papillary_dermis,acanthosis,hyperkeratosis,perekeratosis, clubbing_of_the_rete_ridges,elongation_of_the_rete_ridges,thinning_of_the_suprapapillary_epidermis,spongiform_pustule, munro_microabcess,focal_hypergranulosis,disappearance_of_the_granular_layer,vacuolisation_and_damage_of_basal_layer, saw- tooth_appearance_of_retes,follicular_horn_plug,perifollicular_parakeratosis,band-like_infiltrate they all have highest chance of having class 4 (pityriasis rosea) skin dieases problem.
*   A Patient who has feature not present in erythema,definite border,scaling,itiching,koebner phenomnen,polygonal_papules, follicular_papules,oral_mucosal_involvement,knee_and_elbow_involvement,scalp involvement,melanin_incontinence, eosinophils_in_the_infiltrate,PNL infiltrate,exocytosis,hyperkeratosis,perakeratosis,clubbing_of_the_rete_ridges, thinning_of_the_suprapapillary_epidermis,spongiform_pustule,munro_microabcess,focal_hypergranulosis, disappearance_of_the_granular_layer,vacuolisation_and_damage_of_basal_layer,spongiosis,saw-tooth_appearance_of_retes, folicular horn plug,perifollicular_parakeratosis,band-like_infiltrate they all have highest chance of having class 5 (cronic dermatitis) skin disorder problem.
*   A Patient who has feature not present in definite border,itiching,koebner_phenomenon,polygonal papules,munro_microabcess, oral_mucosal_involvement,scalp involvement,melanin_incontinence,eosinophils_in_the_infiltrate,PNL inftitrate, fibrosis_of_the_papillary_dermis,clubbing_of_the_rete_ridges,elongation_of_the_rete_ridges,spongiform_pustule, thinning_of_the_suprapapillary_epidermis,focal_hypergranulosis,disappearance_of_the_granular_layer,saw- tooth_appearance_of_retes,vacuolisation_and_damage_of_basal_layer,band-like_infiltrate they all have highest chance of having class 6 (pityriasis rubra pilaris) skin disorder problem.
  
#### Intermediate level of skin dieases.
*   A Patient who has intermediate level of erythema,scaling,definite border,knee and elbow envolvement,scalp envolvement,PNL infitrate,acanthosis,hyperkertosis,parakeratosis,clubbing of the rate ridge,elegation of the rate ridge,sponiform pustle,munro microbsess,disapperiance of the granular layer,inflammatory monoluclear inifitrate they all have highest chance of having class 1 (psoriasis) skin disorder problem.
*   A Patient who has intermediate level of erythema,scaling,definite border,itiching,koebner phonomenon,poligonal papules,oral mucosal invlovement,melanin incontience,exocytosis,acanthosis,focal hypergranulosis,vacuolisation and damage of base layer,saw tooth apperance of retes,inflammatory monocular inflitrate they all have highest chance og having class 3 (lichen planus) skin dieases problem.
*   A Patient who has intermediate level of erythema,scaling,itiching,PNLinflatrate,expcytosis,acanthosis,spngosis, inflammatory monoluclear inifitrate they all have highest chance of having class 2 (seboreic dermatitis) skin dieases problem.
*   A Patient who has intermediate level of erythema,scaling,definite border,koebner phonemnon,exocytosis,acanthosis,perakeratosis,spongiosis,inflammatory monoluclear inifitrate they all have highest chance of having class 4 (pityriasis rosea) skin dieases problem.
*   A  Patient who has intermediate level of erythema,itiching,fibrosis of the papilarydermis,aconthosis,hyperkeratosis, perakeratosis,elegation of the rete ridges,inflammatory monoluclear inifitrate they all have highest chance of having class 5 (cronic dermatitis) skin dieases problem.
*   A Patient who has intermediate level of erythema,scaling,folicular papules,knee and elbow envolvement,scalp involvement,acanthosis,hyperkeartosis,folicular horn plug,perifolicular perakeratosis they all have highest chance of having class 6 (pityriasis rubra pilaris) skin dieases problem.

#### High chances of skin dieases.
*   A Patient who have largest amount possible of erythema,scaling,definite border,itiching,knee and elbow envolvement,scalp envolvement,PNL infiltrate,acanthosis,hyperkeratosis,perekeratosis,clubbing_of_the_rete_ridges, elongation_of_the_rete_ridges,thinning_of_the_suprapapillary_epidermis,spongiform_pustule,munro_microabcess, disappearance_of_the_granular_layer,inflammatory_monoluclear_inflitrate they all have highest chance of having class 1 (psoriasis) skin disorder problem.
*   A Patient who have largest amount possible of erythema,scaling,itiching,eosinophils_in_the_infiltrate,PNL infiltrate,exocytosis,acanthosis,spongiosis,inflammatory_monoluclear_inflitrate they all have highest chance of having class 2 (seboreic dermatitis) skin disorder problem.
*   A Patient who have largest amount possible of erythema,scaling,definite border,itiching,koebner_phenomenon, polygonal_papules,band-like_infiltrate oral_mucosal_involvement,melanin_incontinence,exocytosis,acanthosis, focal_hypergranulosis,spongiosis,vacuolisation_and_damage_of_basal_layer,saw-tooth_appearance_of_retes, inflammatory_monoluclear_inflitrate they all having highest chance of class 3 (lichen planus) skin disorder problem.
*   A Patient who have largest amount possible of erythema,koebner_phenomenon,exocytosis,spongiosis, inflammatory_monoluclear_inflitrate they all have highest chance of having class 4 (pityriasis rosea) skin disorder problem.
*   A Patient who have largest amount possible of erythema,scaling,definite border,itiching,fibrosis_of_the_papillary_dermis, acanthosis,elongation_of_the_rete_ridges,inflammatory_monoluclear_inflitrate they all have highest chance of having class 5 (cronic dermatitis) skin disorder problem.
*   A Patient who have largest amount possible of erythema,follicular_papules,knee_and_elbow_involvement,exocytosis, spongiosis,follicular_horn_plug,perifollicular_parakeratosis,inflammatory_monoluclear_inflitrate they all have highest chance of having class 6 (pityriasis rubra pilaris) skin disorder problem.
*   Patient whose family has no skin dieases they have high chance of having class 1 and 3, and 50-50% chance of 2,4 and 5 and those patient which family has skin dieases they have high chance of having class 1 and 6.
By follow this all instructions doctors can find any skin disorder of class 6 class qiuckly.

#### Conclusion of Model Comparison Report

*   I have used 6 Algorithmns which name are LogisticRegression,KNeighborsClassifier, Support Vector Classifier, DecisionTreeClassifier,RandomForestClassifier and ANN_MLPClassifier for training the model. I got 98.91 percentage in LogisticRegression and also in RandomForestClassifier which are maximum than all Algorithmn and its working Mindblowing and error rate only 1.09 which are minor error and model predict perfect results. So, I am preffering LogisticRegression and RandomForestClassifier for identify the skin diseases of the patient at the earliest time.
