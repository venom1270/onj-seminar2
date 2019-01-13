from nltk import agreement

f = open("../data/dataset - fixed.csv", "r", encoding="utf-8")

first = True
scores = []
for line in f:
    if first:
        first = False
        continue
    s = line.split(";")
    scores.append((s[12], s[13], s[14], s[11]))

print(scores)

#glenn = [s[0] for s in scores]
#amber = [s[1] for s in scores]
#final = [s[2] for s in scores]
questions = [s[3] for s in scores]

glenn = []
amber = []
final =[]

tdata = []
for i in range(len(scores)):
    glenn.append(('g', scores[i][3], scores[i][0]))
    amber.append(('a', scores[i][3], scores[i][1]))
    final.append(('F', scores[i][3], scores[i][2]))

print(tdata)

print("Inter-agreement")
tdata = glenn+amber
ratingtask = agreement.AnnotationTask(data=tdata)
print("kappa " +str(ratingtask.kappa()))
print("fleiss " + str(ratingtask.multi_kappa()))
print("alpha " +str(ratingtask.alpha()))
print("scotts " + str(ratingtask.pi()))

print("Glenn vs. Final score")
tdata = glenn+final
ratingtask = agreement.AnnotationTask(data=tdata)
print("kappa " +str(ratingtask.kappa()))
print("fleiss " + str(ratingtask.multi_kappa()))
print("alpha " +str(ratingtask.alpha()))
print("scotts " + str(ratingtask.pi()))

print("Amber vs. Final score")
tdata = amber+final
ratingtask = agreement.AnnotationTask(data=tdata)
print("kappa " +str(ratingtask.kappa()))
print("fleiss " + str(ratingtask.multi_kappa()))
print("alpha " +str(ratingtask.alpha()))
print("scotts " + str(ratingtask.pi()))


#taskdata=[[0,str(i),str(rater1[i])] for i in range(0,len(rater1))]+[[1,str(i),str(rater2[i])] for i in range(0,len(rater2))]+[[2,str(i),str(rater3[i])] for i in range(0,len(rater3))]