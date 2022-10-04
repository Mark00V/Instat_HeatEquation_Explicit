import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

# all vars: 
# xi, order, node 

# -----------------------------------
# Lobattozeros
def lobattozeros(order):
    ret = {
        1: [0.0, 1.0],
        2: [0.0, 0.5, 1.0],
        3: [0.0, 0.2763932, 0.7236068, 1.0],
        4: [0.0, 0.17267316, 0.5,0.82732684, 1.0]
    }
    return(ret[order])
# -----------------------------------

# -----------------------------------
# Zeros equidistant
def equzeros(order):
    ret = {
        1: [0.0, 1.0],
        2: [0.0, 0.5, 1.0],
        3: [0.0, 0.33333333, 0.66666666, 1.0],
        4: [0.0, 0.25, 0.5,0.75, 1.0]
    }
    return(ret[order])
# -----------------------------------

# -----------------------------------
# Integration nodes Gauss Legendre
def intnodes_GL(order):
    ret = {
        1: [0.21132487, 0.78867513],
        2: [0.11270167, 0.5, 0.88729833],
        3: [0.069431844, 0.33000948, 0.66999052, 0.93056816],
        4: [0.046910077, 0.23076534, 0.5, 0.76923466, 0.95308992]
    }
    return(ret[order])
# -----------------------------------

# -----------------------------------
# Integration weights Gauss legendre
def intweights_GL(order):
    ret = {
        1: [0.5, 0.5],
        2: [0.27777778, 0.44444444, 0.27777778],
        3: [0.17392742, 0.32607258, 0.32607258, 0.17392742],
        4: [0.11846344, 0.23931434, 0.28444444, 0.23931434, 0.11846344]
    }
    return(ret[order])
# -----------------------------------

# -----------------------------------
# Integration nodes Lobatto
def intnodes_LOB(order):
    ret = {
        1: [0.0, 1.0],
        2: [0.0, 0.5, 1.0],
        3: [0.0, 0.2763932, 0.7236068, 1.0],
        4: [0.0, 0.17267316, 0.5, 0.82732684, 1.0]    
    }
    return(ret[order])
# -----------------------------------

# -----------------------------------
# Integration weights Lobatto
def intweights_LOB(order):
    ret = {
        1: [0.5, 0.5],
        2: [0.16666667, 0.66666667, 0.16666667],
        3: [0.083333333, 0.41666667, 0.41666667, 0.083333333],
        4: [0.05, 0.27222222, 0.35555556, 0.27222222, 0.05]    
    }
    return(ret[order])
# -----------------------------------

# -----------------------------------
# Formfunktionen Equidistant
def philequdist(xi, order, node):

    if order == 1:
        ret = {
            1: 1-xi,
            2: xi
        }
    
    elif order == 2:
        ret = {
            1: -2*(1-xi)*(-(1/2)+xi),
            2: -4*(-1+xi)*xi,
            3: 2*(-(1/2)+xi)*xi
        }

    elif order == 3:
        ret = {        
            1: 9/2*(1-xi)*(-(2/3)+xi)*(-(1/3)+xi),
            2: 27/2*(-1+xi)*(-(2/3)+xi)*xi,
            3: -(27/2)*(-1+xi)*(-(1/3)+xi)*xi,
            4: 9/2*(-(2/3)+xi)*(-(1/3)+xi)*xi
        }

    elif order == 4:
        ret = {  
            1: -(32/3)*(1-xi)*(-(3/4)+xi)*(-(1/2)+xi)*(-(1/4)+xi),
            2: -(128/3)*(-1+xi)*(-(3/4)+xi)*(-(1/2)+xi)*xi,
            3: 64*(-1+xi)*(-(3/4)+xi)*(-(1/4)+xi)*xi,
            4: -(128/3)*(-1+xi)*(-(1/2)+xi)*(-(1/4)+xi)*xi,
            5: 32/3*(-(3/4)+xi)*(-(1/2)+xi)*(-(1/4)+xi)*xi
        }
    return(ret[node])
# -----------------------------------

# -----------------------------------
# Gradient Formfunktionen Equidistant
def philequdistgrad(xi, order, node):
    
    if order == 1:
        ret = {
            1: -1,
            2: 1
        }

    elif order == 2:
        ret = {
            1: -2*(1-xi)+2*(-(1/2)+xi),
            2: -4*(-1+xi)-4*xi,
            3: 2*(-(1/2)+xi)+2*xi
        }

    elif order == 3:
        ret = {
            1: 9/2*(1-xi)*(-(2/3)+xi)+9/2*(1-xi)*(-(1/3)+xi)-9/2*(-(2/3)+xi)*(-(1/3)+xi),
            2: 27/2*(-1+xi)*(-(2/3)+xi)+27/2*(-1+xi)*xi+27/2*(-(2/3)+xi)*xi,
            3: -(27/2)*(-1+xi)*(-(1/3)+xi)-27/2*(-1+xi)*xi-27/2*(-(1/3)+xi)*xi,
            4: 9/2*(-(2/3)+xi)*(-(1/3)+xi)+9/2*(-(2/3)+xi)*xi+9/2*(-(1/3)+xi)*xi
        }

    elif order == 4:
        ret = {
            1: -(32/3)*(1-xi)*(-(3/4)+xi)*(-(1/2)+xi)-32/3*(1-xi)*(-(3/4)+xi)*(-(1/4)+xi) \
               -32/3*(1-xi)*(-(1/2)+xi)*(-(1/4)+xi)+32/3*(-(3/4)+xi)*(-(1/2)+xi)*(-(1/4)+xi),
            2: -(128/3)*(-1+xi)*(-(3/4)+xi)*(-(1/2)+xi)-128/3*(-1+xi)*(-(3/4)+xi)*xi-128/3 \
               *(-1+xi)*(-(1/2)+xi)*xi-128/3*(-(3/4)+xi)*(-(1/2)+xi)*xi,            
            3: 64*(-1+xi)*(-(3/4)+xi)*(-(1/4)+xi)+64*(-1+xi)*(-(3/4)+xi)*xi+64*(-1+xi) \
               *(-(1/4)+xi)*xi+64*(-(3/4)+xi)*(-(1/4)+xi)*xi,            
            4: -(128/3)*(-1+xi)*(-(1/2)+xi)*(-(1/4)+xi)-128/3*(-1+xi)*(-(1/2)+xi)*xi-128/3 \
               *(-1+xi)*(-(1/4)+xi)*xi-128/3*(-(1/2)+xi)*(-(1/4)+xi)*xi,            
            5: 32/3*(-(3/4)+xi)*(-(1/2)+xi)*(-(1/4)+xi)+32/3*(-(3/4)+xi)*(-(1/2)+xi)*xi+32/3 \
               *(-(3/4)+xi)*(-(1/4)+xi)*xi+32/3*(-(1/2)+xi)*(-(1/4)+xi)*xi
        }
    return(ret[node])
# -----------------------------------

# -----------------------------------
# Formfunktionen Lobatto
def phillobatto(xi, order, node):
    
    if order == 1:
        ret = {
            1: -1.0*(-1.0+xi),
            2: 1.0*xi
        }

    elif order == 2:
        ret = {
            1: 2.0*(-1.0+xi)*(-0.5+xi),
            2: -4.0*(-1.0+xi)*xi,
            3: 2.0*(-0.5+xi)*xi
        }

    elif order == 3:
        ret = {
            1: -5.0*(-1.0+xi)*(-0.72360679774997896964+xi)*(-0.27639320225002103036+xi),
            2: 11.180339887498948482*(-1.0+xi)*(-0.72360679774997896964+xi)*xi,
            3: -11.180339887498948482*(-1.0+xi)*(-0.27639320225002103036+xi)*xi,
            4: 5.0*(-0.72360679774997896964+xi)*(-0.27639320225002103036+xi)*xi
        }
        
    elif order == 4:
        ret = {
            1: 14.0*(-1.0+xi)*(-0.82732683535398857190+xi)*(-0.5+xi)*(-0.17267316464601142810+xi),
            2: -32.666666666666666667*(-1.0+xi)*(-0.82732683535398857190+xi)*(-0.5+xi)*xi,
            3: 37.333333333333333333*(-1.0+xi)*(-0.82732683535398857190+xi)*(-0.17267316464601142810+xi)*xi,
            4: -32.666666666666666667*(-1.0+xi)*(-0.5+xi)*(-0.17267316464601142810+xi)*xi,
            5: 14.0*(-0.82732683535398857190+xi)*(-0.5+xi)*(-0.17267316464601142810+xi)*xi
        }
    return(ret[node])
# -----------------------------------

# -----------------------------------
# Gradient Formfunktionen Lobatto
def phillobattograd(xi, order, node):
    
    if order == 1:
        ret = {
            1: -1.0,
            2: 1.0
        }
        
    elif order == 2:
        ret = {
            1: 2.0*(-1.0+xi)+2.0*(-0.50+xi),
            2: -4.0*(-1.0+xi)-4.0*xi,
            3: 2.0*(-0.50+xi)+2.0*xi
        }

    elif order == 3:
        ret = {
            1: -5.0*(-1.0+xi)*(-0.72360679774997896964+xi)-5.0*(-1.0+xi)*(-0.27639320225002103036+xi) \
               -5.0*(-0.72360679774997896964+xi)*(-0.27639320225002103036+xi),
            2: 11.180339887498948482*(-1.0+xi)*(-0.72360679774997896964+xi)+11.180339887498948482*(-1.0+xi) \
               *xi+11.180339887498948482*(-0.72360679774997896964+xi)*xi,
            3: -11.180339887498948482*(-1.0+xi)*(-0.27639320225002103036+xi)-11.180339887498948482*(-1.0+xi) \
               *xi-11.180339887498948482*(-0.27639320225002103036+xi)*xi,
            4: 5.0*(-0.72360679774997896964+xi)*(-0.27639320225002103036+xi)+5.0*(-0.72360679774997896964+xi) \
               *xi+5.0*(-0.27639320225002103036+xi)*xi
        }
    
    elif order == 4:
        ret = {
            1: 14.0*(-1.0+xi)*(-0.82732683535398857190+xi)*(-0.50+xi)+14.0*(-1.0+xi)*(-0.82732683535398857190+xi) \
               *(-0.17267316464601142810+xi)+14.0*(-1.0+xi)*(-0.50+xi)*(-0.17267316464601142810+xi) \
               +14.0*(-0.82732683535398857190+xi)*(-0.50+xi)*(-0.17267316464601142810+xi),
            2: -32.666666666666666667*(-1.0+xi)*(-0.82732683535398857190+xi)*(-0.50+xi)-32.666666666666666667 \
               *(-1.0+xi)*(-0.82732683535398857190+xi)*xi-32.666666666666666667*(-1.0+xi)*(-0.50+xi) \
               *xi-32.666666666666666667*(-0.82732683535398857190+xi)*(-0.50+xi)*xi,
            3: 37.333333333333333333*(-1.0+xi)*(-0.82732683535398857190+xi)*(-0.17267316464601142810+xi) \
               +37.333333333333333333*(-1.0+xi)*(-0.82732683535398857190+xi)*xi+37.333333333333333333*(-1.0+xi) \
               *(-0.17267316464601142810+xi)*xi+37.333333333333333333*(-0.82732683535398857190+xi) \
               *(-0.17267316464601142810+xi)*xi,
            4: -32.666666666666666667*(-1.0+xi)*(-0.50+xi)*(-0.17267316464601142810+xi)-32.666666666666666667 \
               *(-1.0+xi)*(-0.50+xi)*xi-32.666666666666666667*(-1.0+xi)*(-0.17267316464601142810+xi) \
               *xi-32.666666666666666667*(-0.50+xi)*(-0.17267316464601142810+xi)*xi,
            5: 14.0*(-0.82732683535398857190+xi)*(-0.50+xi)*(-0.17267316464601142810+xi)+14.0 \
               *(-0.82732683535398857190+xi)*(-0.50+xi)*xi+14.0*(-0.82732683535398857190+xi)*(-0.17267316464601142810+xi) \
               *xi+14.0*(-0.50+xi)*(-0.17267316464601142810+xi)*xi
        }
    return(ret[node])
# -----------------------------------

# -----------------------------------
# Berechne Elementmatrizen, falls keine Ordnung m angegeben -> m = 2
# method = "GL -> Gauss Legendre Integration und Equidistante Formfunktionen
# method = "LOB" -> Lobatto Integration und Lobatto Formfunktionen
def elemat(length, k, rho, cp, method = "GL", order = None):
    if order is None:
        m = 2
    else:
        m = order
        
    if method == "GL":
        intnodes = intnodes_GL(m)
        intweights = intweights_GL(m)
        phigrad = philequdistgrad
        phi = philequdist
    elif method == "LOB":
        intnodes = intnodes_LOB(m)
        intweights = intweights_LOB(m)
        phigrad = phillobattograd
        phi = phillobatto       
    
    elesteifmat = np.zeros((m+1,m+1),dtype=np.double)
    for j in range(0,m+1):
        val = [[1/length*phigrad(intnodes[j],m,i+1)*phigrad(intnodes[j],m,ii+1)*intweights[j] for ii in range(0,m+1)] for i in range(0,m+1)]
        elesteifmat += val
    elesteifmat *= k
    
    elemassmat = np.zeros((m+1,m+1),dtype=np.double)
    for j in range(0,m+1):
        val = [[length*phi(intnodes[j],m,i+1)*phi(intnodes[j],m,ii+1)*intweights[j] for ii in range(0,m+1)] for i in range(0,m+1)]
        elemassmat += val
    elemassmat *= rho*cp
    
    if method == "LOB":
        elemassmath = np.zeros((m+1,m+1),dtype=np.double)
        np.fill_diagonal(elemassmath,elemassmat.diagonal())
        elemassmat = elemassmath

    return(elesteifmat, elemassmat)    
# -----------------------------------        

# -----------------------------------   
# Assembliere Systemmatrix
# vector_mats enthält Einzelmatrizen (bspw. Elementmatrizen)
# Jeweils End und Anfangsfreiheitsgrad werden verbunden
# Hinweis: effizienter mit Sparsematrizen, hier aber aufgrund geringer Größe nicht notwendig
def assemble(vector_mats):
    nr_of_mats = len(vector_mats)
    length_sysmat = 0
    for mat in vector_mats:
        length_sysmat += len(mat)-1
    length_sysmat +=1
    
    sysmat = np.zeros((length_sysmat,length_sysmat),dtype=np.double)
    pos_a = 0
    pos_e = len(vector_mats[0])
    for mat in vector_mats:
        sysmat[pos_a:pos_e,pos_a:pos_e] += mat
        pos_a += len(mat)-1
        pos_e += len(mat)-1
    return(sysmat)
# -----------------------------------   

# -----------------------------------   
# Zeitschleife
def timeloop(mmat, kmat, init, dt, ndt, vector_rbs, method="GL"):
    gammal = vector_rbs[0]
    gl = vector_rbs[1]
    gammar = vector_rbs[2]
    gr = vector_rbs[3]
    
    # Bei Einbau Randbedingungen in Systemmassenmatrix
    # MAYBE HIER EIN FEHLER???? Muss rsvek mit mmat oder mmatred erstellt werde???
    mmatred = mmat.copy()
    mmatred[0,0] -= gammal
    mmatred[-1,-1] -= gammar
    
    fhg = len(mmat)
    pmat = init*np.ones(fhg)
    for i in tqdm(range(ndt-1), total=ndt-1, unit=""):
        rsvek = (mmat-dt*kmat)@pmat
        
        # Einbau RBs in Lastvektor
        rsvek[0] -= gl
        rsvek[-1] -= gr
        
        if method == "GL":
            sol = np.linalg.solve(mmatred, rsvek)
        elif method == "LOB":
            sol = rsvek/mmatred.diagonal()
        pmat = sol
    return(sol)

# -----------------------------------

# -----------------------------------
# Grafische Ausgabe
def plotsol(sol, allnodes):
    x = allnodes 
    y = sol

    # set the font 
    font_globally = "serif" 
    plt.rcParams.update({'font.family':font_globally})

    # font size
    plt.rc('font', size=12) #controls default text size
    plt.rc('axes', titlesize=12) #fontsize of the title
    plt.rc('axes', labelsize=10) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=10) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=10) #fontsize of the y tick labels
    plt.rc('legend', fontsize=10) #fontsize of the legend

    fig, ax = plt.subplots()
    line,=ax.plot(x, y,"o",color="black") # "o" setzt marker

    ax.set(xlabel='x [m]', ylabel='Temperature [°C]',
    title='Transient temperature')
    ax.grid()
    line.set_dashes([2, 2, 10, 2])
    plt.show()

    print("T at x=0:",y[0],"T at x=",x[-1],y[-1]) 
# -----------------------------------

# -----------------------------------
# Input Parameter
def inputparam():
    print("===============================")
    print("-----------Eingabe-------------")
    lst_rods = {}
    n_rods = int(input("Anzahl der Schichten:"))
    for i in range(n_rods):
        print("------------------------")
        print("Parameter für Schicht", i+1)
        rod_len = float(input("Dicke:"))
        rod_k = float(input("K:"))
        rod_rho = float(input("Rho:"))
        rod_cp = float(input("Cp:"))
        rod_ne = int(input("Anzahl Elemente/Schicht:"))
        lst_rods[i+1] = [rod_len, rod_k, rod_rho, rod_cp, rod_ne]
        print("Schicht ",i,":", "d=",rod_len,"k=",rod_k,"rho=",rod_rho,"Cp=",rod_cp,"ne=",rod_ne)
    print("------------------------")

    print("-------BCs und IC-------")
    tl = float(input("Temperatur links:"))
    hl = float(input("Widerstand links:"))
    if hl != 0:
        hl = 1/hl
    elif hl == 0:
        hl =1/0.0001
    tr = float(input("Temperatur rechts:"))
    hr = float(input("Widerstand rechts:"))
    if hr != 0:
        hr = 1/hr
    elif hr == 0:
        hr =1/0.0001
    print("Links: T=",tl,"h=",1/hl," Rechts: T=",tr,"h=",1/hr)
    gammal = -hl
    gl = -tl*hl
    gammar = -hr
    gr = -tr*hr
    vector_rbs = [gammal, gl, gammar, gr]
    print("------------------------")

    print("--Berechnungsparameter:-")
    # Berechnungsparameter
    method = str(input("LOB oder GL:"))
    order = int(input("Ordung Formfunktionen:"))

    # Zeitdiskretisierung
    init = float(input("Anfangsbedingung:"))
    dt = float(input("Zeitschrittweite:"))
    ndt = int(input("Anzahl Zeitschritt:"))
    print("Endzeitpunkt=",dt*ndt,"s")
    print("------------------------")
    print("===============================")
    print("-------Starte Berechnung-------")
    
    return(lst_rods, vector_rbs, method, order, init, dt, ndt)
# -----------------------------------

# -----------------------------------
# Main
def main():

    lst_rods, vector_rbs, method, order, init, dt, ndt = inputparam()
    
    # Berechne Knotenpositionen...geht sicherlich auch eleganter...WIP
    # Berechne Elementmatrizen
    lst_elesteif = [] 
    lst_elemass = [] 
    allnodes = [0]
    pos = 0
    for i in range(1,len(lst_rods)+1):
        len_rod = lst_rods[i][0]
        n_elements_rod = lst_rods[i][4]
        
        len_elements = len_rod/n_elements_rod
        k = lst_rods[i][1]
        rho = lst_rods[i][2]
        cp = lst_rods[i][3]
        print("len_elements=",len_elements,"k=",k,"rho=",rho,"cp=",cp,"n_elements_rod=",n_elements_rod)
        
        for nele in range(n_elements_rod):
            elemats = elemat(len_elements, k, rho, cp, method, order)
            lst_elesteif.append(elemats[0])
            lst_elemass.append(elemats[1])            
        
        if method == "GL":
            lstnodes = equzeros
        elif method == "LOB":
            lstnodes = lobattozeros
        lstdx = np.array(lstnodes(order))*len_elements
        for n in range(n_elements_rod):
            for i in range(len(lstdx)-1):
                dx = lstdx[i+1]-lstdx[i]
                pos += dx
                allnodes.append(pos)
                
    # Systemmatrizen
    kmat = assemble(lst_elesteif)
    mmat = assemble(lst_elemass)
    
    # Zeitschleife
    sol = timeloop(mmat, kmat, init, dt, ndt, vector_rbs)
    
    # Ausgabe Plot
    plotsol(sol, allnodes)
# -----------------------------------

main()