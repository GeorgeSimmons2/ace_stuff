using ExtXYZ #ACEpotentials, ExtXYZ

training_data = ExtXYZ.load("./high_entropy_pops/df_train_Al.xyz")
println(training_data[1])