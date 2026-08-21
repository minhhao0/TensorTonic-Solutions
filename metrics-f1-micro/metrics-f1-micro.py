def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Return the micro-averaged F1 score rounded to four decimals.
    """
    # Write code here
    classes=set(y_true)
    tp=0
    fp=0
    fn=0
    for c in classes:
        for i in range(len(y_pred)):
            if y_pred[i]==c:
                if y_true[i]==y_pred[i]:
                    tp+=1
                if y_true[i]!=y_pred[i]:
                    fp+=1
            else:
                if y_true[i]==c:
                    fn+=1
    score=(2*tp)/(2*tp+fp+fn)
    return score
        