#!/usr/bin/env python3
# vim: ts=2 sw=2 sts=2 et :

#                 .        +          .      .          .
#          .            _        .                    .
#       ,              /;-._,-.____        ,-----.__
#      ((        .    (_:#::_.:::. `-._   /:, /-._, `._,
#       `                 \   _|`"=:_::.`.);  \ __/ /
#                           ,    `./  \:. `.   )==-'  .
#         .      ., ,-=-.  ,\, +#./`   \:.  / /           .
#     .           \/:/`-' , ,\ '` ` `   ): , /_  -o
#            .    /:+- - + +- : :- + + -:'  /(o-) \)     .
#       .      ,=':  \    ` `/` ' , , ,:' `'--".--"---._/`7
#        `.   (    \: \,-._` ` + '\, ,"   _,--._,---":.__/
#                   \:  `  X` _| _,\/'   .-'
#     .               ":._:`\____  /:'  /      .           .
#                         \::.  :\/:'  /              +
#        .                 `.:.  /:'  }      .
#                .           ):_(:;   \           .
#                           /:. _/ ,  |
#                        . (|::.     ,`                  .
#          .                |::.    {\
#                           |::.\  \ `.
#                           |:::(\    |
#                   O       |:::/{ }  |                  (o
#                    )  ___/#\::`/ (O "==._____   O, (O  /`
#               ~~~w/w~"~~,\` `:/,-(~`"~~~~~~~~"~o~\~/~w|/~
#     dew   ~~~~~~~~~~~~~~~~~~~~~~~\\W~~~~~~~~~~~~\|/~~

"""
./sublime.py [OPTIONS]  
(c)2022 Tim Menzies, unlicense.org.    
Look, a little, before you leap, a lot. Random projections
for bi-clustering.  Iterative dichotomization using ranges
that most distinguish sibling clusters. Repeat, recursively.
Use results for various knowledge-level tasks.

OPTIONS:  

    -Max    max numbers to keep         : 512  
    -Some   find `far` in this many egs : 512  
    -data   data file                   : ../data/auto93.csv   
    -enough min leaf size               : .5
    -help   show help                   : False  
    -far    how far to look in `Some`   : .9  
    -p      distance coefficient        : 2  
    -seed   random number seed          : 10019  
    -todo   start up task               : nothing  
    -xsmall Cohen's small effect        : .35  
"""

import re,sys,random

# This is free and unencumbered software released into the public domain.
#
# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.
#
# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# For more information, please refer to <http://unlicense.org/>

#  ___              __        
# /\_ \      __    /\ \       
# \//\ \    /\_\   \ \ \____  
#   \ \ \   \/\ \   \ \ '__`\ 
#    \_\ \_  \ \ \   \ \ \L\ \
#    /\____\  \ \_\   \ \_,__/
#    \/____/   \/_/    \/___/ 
#-------------------------------------------------------------------------------

# randoms stuff
r        = random.random
anywhere = lambda a: random.randint(0, len(a)-1)

# useful constants
big      = sys.maxsize

# list membership
first    = lambda a: a[0]
second   = lambda a: a[1]

def atom(x):
  "Return a number or trimmed string."
  x=x.strip()
  if   x=="True" : return True
  elif x=="False": return False
  else: 
    try: return int(x)
    except:
      try: return float(x)
      except: return x.strip()
  
def demo(want,all): 
  "Maybe run a demo, if we want it, resetting random seed first."
  for one in dir(all):
    if (not want or (want and one.startswith(want))):
      random.seed(the.seed)
      all.__dict__[one]()

def file(f):
  "Iterator. Returns one row at a time, as cells."
  with open(f) as fp:
    for line in fp: 
      line = re.sub(r'([\n\t\r"\' ]|#.*)', '', line)
      if line:
        yield [atom(cell.strip()) for cell in line.split(",")]

def merge(b4):
  j,n,tmp = -1,len(b4),[]
  while j < n-1:
    j += 1
    a = b4[j]
    if j < n-2:
      if b := a.merge(b4[j+1]):
        a = b
        j += 1
    tmp += [a]
  return b4 if len(b4)==len(all) else merge(tmp)  

class o(object):
  "Class that can pretty print its slots, with fast init."
  def __init__(i, **d): i.__dict__.update(**d)
  def __repr__(i): 
    pre = i.__class__.__name__ if issubclass(i,o) else ""
    return pre+str(
      {k: v for k, v in sorted(i.__dict__.items()) if str(k)[0] != "_"})

def options(doc):
  "Convert __doc__ string to options directory."
  d={}
  for line in doc.splitlines():
    if line and line.startswith("    -"):
       key, *_, x = line.strip()[1:].split(" ") # get 1st,last word on each line
       for j,flag in enumerate(sys.argv):
         if flag and flag[0]=="-" and key.startswith(flag[1:]):
           x= "True" if x=="False" else("False" if x=="True" else sys.argv[j+1])
       d[key] = atom(x)
  if d["help"]: exit(print(doc))
  return o(**d)

the = options(__doc__)
#          ___                                                
#         /\_ \                                               
#   ___   \//\ \       __       ____    ____     __     ____  
#  /'___\   \ \ \    /'__`\    /',__\  /',__\  /'__`\  /',__\ 
# /\ \__/    \_\ \_ /\ \L\.\_ /\__, `\/\__, `\/\  __/ /\__, `\
# \ \____\   /\____\\ \__/.\_\\/\____/\/\____/\ \____\\/\____/
#  \/____/   \/____/ \/__/\/_/ \/___/  \/___/  \/____/ \/___/ 
#-------------------------------------------------------------------------------

class Span(o):
  "Track the `y` symbols seen in the range `lo` to `hi`."
  def __init__(i,col, lo, hi, ys=None,):
    i.col, i.lo, i.hi, i.B, i.R, i.ys = col, lo, hi,  ys or Sym()

  def add(i,x,y, inc=1):
    i.lo = min(x,i.lo)
    i.hi = max(x,i.hi)
    i.ys.add(y,inc)

  def __lt__(i,j):
    s1, e1 = i.ys.n / i.col.n, i.ys.div()
    s2, s2 = j.ys.n / j.col.n, j.ys.div()
    return ((1 - s1)**2 + e1**2)**.5 < ((1 - s2)**2 + e2**2)**.5

  def merge(i,j): 
    a, b, c = i.ys, j.ys, i.ys.merge(j.ys)
    if c.div()*.99 <= (a.n*a.div() + b.n*b.div())/(a.n + b.n): 
      return Span(i.col, min(i.lo,j.lo),max(i.hi,j.hi), ys=c) 

  def __repr__(i):
    if i.lo == i.hi: return f"{i.col.txt} == {i.lo}"
    if i.lo == -big: return f"{i.col.txt} == {i.hi}"
    if i.hi ==  big: return f"{i.col.txt} >= {i.lo}"
    return f"{i.lo} <= {i.col.txt} < {i.hi}"

  def selects(i,row):
    x = row[col.at]; return x=="?" or i.lo<=x and x<i.hi 

class Col(o):
  "Summarize columns."
  def __init__(i,at=0,txt=""): 
    i.n,i.at,i.txt,i.w=0,at,txt,(-1 if "<" in txt else 1)

  def dist(i,x,y): 
    return 1 if x=="?" and y=="?" else i.dist1(x,y)
    
class Num(Col):
  "Summarize numeric columns."
  def __init__(i,**kw):
    super().__init__(**kw)
    i._all, i.lo, i.hi, i.max, i.ok = [], 1E32, -1E32, the.Max, False

  def add(i,x,_):
    if x != "?":
      i.n += 1
      i.lo = min(x,i.lo)
      i.hi = max(x,i.hi)
      if len(i._all) < i.max    : i.ok=False; i._all += [x]
      elif r()       < i.max/i.n: i.ok=False; i._all[anywhere(i._all)] = x 
    return x

  def all(i):
    if not i.ok: i.ok=True; i._all.sort()
    return i._all

  def per(i,p=.5): 
    a = i.all(); return a[ int(p*len(a)) ]

  def mid(i): return i.per(.5)
  def div(i): return (i.per(.9) - i.per(.1)) / 2.56

  def merge(i,j):
    k = Num(at=i.at, txt=i.txt)
    for x in i._all: k.add(x)
    for x in j._all: k.add(x)
    return k

  def norm(i,x):
    return 0 if i.hi-i.lo < 1E-9 else (x-i.lo)/(i.hi-i.lo)

  def dist1(i,x,y):
    if   x=="?": y=i.norm(y); x=(1 if y<.5 else 0)
    elif y=="?": x=i.norm(x); y=(1 if x<.5 else 0)
    else       : x,y = i.norm(x), i.norm(y)
    return abs(x-y)

  def spans(i,j, all):
    lo  = min(i.lo, j.lo)
    hi  = max(i.hi, j.hi)
    gap = (hi-lo) / (6/the.xsmall)
    at  = lambda z: lo + int((z-lo)/gap)*gap 
    tmp = {}
    for x in map(at, i._all): 
      s = tmp[x] = tmp[x] if x in tmp else Span(i,x,x+gap)
      s.add(x,0)
    for x in map(at, j._all): 
      s = tmp[x] = tmp[x] if x in tmp else Span(i,x,x+gap)
      s.add(x,1)
    tmp = merge([x for _,x in sorted(tmp.items(),key=first)])
    if len(tmp) > 1 : all + tmp

class Sym(Col):
  "Summarize symbolic columns."
  def __init__(i,**kw):
    super().__init__(**kw)
    i.has, i.mode, i.most = {}, None, 0

  def add(i,x,inc):
    if x != "?":
      i.n += inc
      tmp = i.has[x] = inc + i.has.get(x,0)
      if tmp > i.most: i.most, i.mode = tmp, x
    return x

  def dist(i,x,y): return 0 if x==y else 1

  def div(i): 
    p = lambda x: x/i.n
    return sum( -p(x)*math.log(p(x),2) for x in i.has.values() )

  def mid(i): return i.mode

  def merge(i,j):
    k = Sym(at=i.at, txt=i.txt)
    for k,n in i.has.items(): k.add(x,n)
    for k,n in j.has.items(): k.add(x,n)
    return k

  def spans(i,j, all):
    tmp = {}
    for x,n in i.has.items(): 
      s = tmp[x] = (tmp[x] if x in tmp else Span(i,x,x))
      s.add(x,0,n)
    for x,n in j.has.items(): 
      s = tmp[x] = (tmp[x] if x in tmp else Span(i,x,x))
      s.add(x,1,n)
    tmp = [second(x) for x in sorted(tmp.items(), key=first)]
    if len(tmp) > 1 : all + tmp








#-------------------------------------------------------------------------------

class Sample(o):
  "Load, then manage, a set of examples."
  def __init__(i,inits=[]): 
    i.rows, i.cols, i.x, i.y = [], [], [], []
    if str ==type(inits): [i + row for row in file(inits)]
    if list==type(inits): [i + row for row in inits]

  def __add__(i,a):
    def col(at,txt):
      what  = Num if txt[0].isupper() else Sym
      now   = what(at=at, txt=txt)
      where = i.y if "+" in txt or "-" in txt or "!" in txt else i.x
      if txt[-1] != ":": where += [now]
      return now
    #----------- 
    if i.cols: i.rows += [[col.add(a[col.at]) for col in i.cols]]
    else:      i.cols  = [col(at,txt) for at,txt in enumerate(a)]

  def mid(i,cols=None): return [col.mid() for col in (cols or i.all)]
  def div(i,cols=None): return [col.div() for col in (cols or i.all)]

  def clone(i,inits=[]):
    out = Sample()
    out + [col.txt for col in i.cols]
    [out + x for x in inits]
    return out 

  def dist(i,x,y):
    d = sum( col.dist(x[col.at], y[col.at])**the.p for col in i.x )
    return (d/len(i.x)) ** (1/the.p)

  def far(i, x, rows=None):
    tmp= sorted([(i.dist(x,y),y) for y in (rows or i.rows)],key=first)
    return tmp[ int(len(tmp)*the.far) ]

  def proj(i,row,x,y,c):
    a = i.dist(row,x)
    b = i.dist(row,y)
    return (a**2 + c**2 - b**2) / (2*c) , row

  def half(i, top=None):
    top  = top or i
    some = random.choices(i.rows, k=the.Some)
    w    = some[0]
    _,x  = top.far(w, some)
    c,y  = top.far(x, some)
    left, right = i.clone(), i.clone()
    for n,(_,r) in enumerate(
                     sorted([top.proj(r,x,y,c) for r in i.rows],key=first)):
      (left if n <= len(i.rows)//2 else right).__add__(r) 
    return left,right

  def split(i,top=None):
    top = top or i
    if len(i.rows) < 2*len(top.rows)**the.enough:
      return i
    left0, right0 = i.half(top)
    spans = []
    for lcol,rcol in zip(left0.x, right0.x):
      lcol.spans(rcol,spans)
    span = sorted(spans)[0]
    left, right = i.clone(), i.clone()
    for row in i.rows:
      (left if span.selects(row) else right).add(row)
    return o(here=i, when=span, left=left.split(top), right=right.split(top))
      
#   _
#  /\ \                                         
#  \_\ \      __     ___ ___      ___     ____  
#  /'_` \   /'__`\ /' __` __`\   / __`\  /',__\ 
# /\ \L\ \ /\  __/ /\ \/\ \/\ \ /\ \L\ \/\__, `\
# \ \___,_\\ \____\\ \_\ \_\ \_\\ \____/\/\____/
#  \/__,_ / \/____/ \/_/\/_/\/_/ \/___/  \/___/ 
#------------------------------------------------------------------------------

class Demos:
  "Possible start-up actions."
  def num(): 
    n=Num()
    for x in range(10000): n.add(x)
    print(sorted(n._all),n)

  def sym(): 
    s=Sym()
    for x in range(10000): s.add( int(r()*20))
    print(s)

  def rows(): 
    for row in file(the.data): print(row)

  def sample(): s=Sample(the.data); print(len(s.rows))

  def done(): s=Sample(the.data); s.dist(s.rows[1], s.rows[2])

  def dist():
    s=Sample(the.data)
    for row in s.rows: print(s.dist(s.rows[0], row))

  def far():
    s=Sample(the.data)
    for row in s.rows: print(row,s.far(row))

  def clone():
    s=Sample(the.data); s1 = s.clone(s.rows)
    print(s.x[0])
    print(s1.x[0])

  def half():
    s=Sample(the.data); s1,s2 = s.half()
    print(s1.mid(s1.y))
    print(s2.mid(s2.y))

if __name__ == "__main__": 
  demo(the.todo,Demos)
