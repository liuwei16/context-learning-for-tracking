function op = overlap(x1, w1, x2, w2)
        l1 = x1 - w1/2;
        l2 = x2 - w2/2;
        if l1 > l2, left = l1; else left = l2 ; end
        r1 = x1 + w1/2;
        r2 = x2 + w2/2;
        if r1 < r2, right = r1;  else right = r2 ; end
        op = right - left;
   end