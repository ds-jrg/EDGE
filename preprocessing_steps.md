# Manual pre.processing of the knowledge Graphs

The Knowledge Graphs have been created using the same data scources i.e, files from DGL distribution and are readily provided in the framework, which can be used to start  and run explainers. They can also be recreated using the followings steps
## First run the KG preprocessor file:
```shell
python preprocess_kg.py
```

After running the preprocessor script, navigate to the directory containing the Knowledge Graphs (KGs) with the following command:

```shell
cd data/KGs
```

## Converting the N3/NT files to OWL format

```shell
robot convert --input aifbfixed_complete_processed.n3 --output aifb.owl
robot convert --input mutag_stripped_processed.nt --output mutag.owl

```

## Preprocessing BGS dataset

Navigate to the folder containing BGS dataset :
```shell
cd ../bgs-hetero_733c98ba
```
In the terminal, run the following command.
``` shell
robot merge --input Lexicon_SourceInfo.nt --input Geochronology_DivisionList.nt --input Lexicon_Class.nt --input Geochronology_Scheme.nt --input Lexicon_LithologyComponent.nt --input Lexicon_SpatialScope.nt --input EarthMaterialClass.nt --input EarthMaterialClass_RockDummy.nt --input EarthMaterialClass_RockComposite.nt --input Lexicon_LithogeneticType.nt --input Lexicon_DefinitionStatus.nt --input Geochronology.nt --input Lexicon_NamedRockUnit.nt --input Lexicon_RockUnitRank.nt --input Geochronology_AgeDeterminationType.nt --input Geochronology_RankList.nt --input EarthMaterialClass_RockComponent.nt --input Lexicon_EquivalentName.nt --input EarthMaterialClass_ComponentRank.nt --input 625KGeologyMap_Unit.nt --input 625KGeologyMap_Rank.nt --input 625KGeologyMap.nt --input Lexicon.nt --input Lexicon_Stratotype.nt --input Lexicon_StratotypeType.nt --input Spatial.nt --input 625KGeologyMap_Fault.nt --input 625KGeologyMap_Dyke.nt --input Lexicon_Theme.nt --input EarthMaterialClass_RockName.nt --input Geochronology_Boundary.nt --input EarthMaterialClass_ComponentRelation.nt --input Lexicon_EquivalenceType.nt --input Lexicon_ShapeType.nt --input Geochronology_Division.nt --input Geochronology_Rank.nt --output bgs_preprocessed.owl
```

Then use the ROBOT library to repair the BGS OWL file.
```shell
robot repair --input bgs_preprocessed.owl --output bgs_repaired.owl
```

Then, move to the root directory of the project using the command:
```shell
cd ../..
```
And finally complete the dataset preprocessing using the command:
```shell
python postprocess_kg.py
```