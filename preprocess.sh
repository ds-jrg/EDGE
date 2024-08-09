echo " Starting Preprocessing script.. !!"

# First run the KG preprocessor file:
python preprocess_kg.py

echo "Navigate into Knowledge Graph Directory"
cd data/KGs

robot convert --input aifbfixed_complete_processed.n3 --output aifb.owl
robot convert --input mutag_stripped_processed.nt --output mutag.owl


echo "Navigate to the folder containing BGS dataset :"
cd ../bgs-hetero_733c98ba

echo "Use the ROBOT library to merge all BGS nt files and repair the file for possible errors during merging."
robot merge --input Lexicon_SourceInfo.nt --input Geochronology_DivisionList.nt --input Lexicon_Class.nt --input Geochronology_Scheme.nt --input Lexicon_LithologyComponent.nt --input Lexicon_SpatialScope.nt --input EarthMaterialClass.nt --input EarthMaterialClass_RockDummy.nt --input EarthMaterialClass_RockComposite.nt --input Lexicon_LithogeneticType.nt --input Lexicon_DefinitionStatus.nt --input Geochronology.nt --input Lexicon_NamedRockUnit.nt --input Lexicon_RockUnitRank.nt --input Geochronology_AgeDeterminationType.nt --input Geochronology_RankList.nt --input EarthMaterialClass_RockComponent.nt --input Lexicon_EquivalentName.nt --input EarthMaterialClass_ComponentRank.nt --input 625KGeologyMap_Unit.nt --input 625KGeologyMap_Rank.nt --input 625KGeologyMap.nt --input Lexicon.nt --input Lexicon_Stratotype.nt --input Lexicon_StratotypeType.nt --input Spatial.nt --input 625KGeologyMap_Fault.nt --input 625KGeologyMap_Dyke.nt --input Lexicon_Theme.nt --input EarthMaterialClass_RockName.nt --input Geochronology_Boundary.nt --input EarthMaterialClass_ComponentRelation.nt --input Lexicon_EquivalenceType.nt --input Lexicon_ShapeType.nt --input Geochronology_Division.nt --input Geochronology_Rank.nt --output bgs_preprocessed.owl

robot repair --input bgs_preprocessed.owl --output bgs_repaired.owl

cd ../..

echo "Run the postprocessing script for creating KGs to train logical explainers."
python postprocess_kg.py