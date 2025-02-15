import numpy as np                                                         # numpy
import matplotlib.pyplot as plt                                            # graficos
import scipy.stats as stats                                                # pvalor y chi2
from scipy.stats import chi2                                               # chi2
import scipy.special                                                       # funciones raras
from scipy.optimize import curve_fit                                       # curv_fit (ajustes)
from scipy.optimize import minimize                                        # minimize (ajustes con metodos)
from scipy.signal import find_peaks                                        # máximos
from scipy.signal import argrelmin                                         # mínimos
import sympy as sp                                                         # sympy 
import pandas as pd

from scipy.optimize._numdiff import approx_derivative
from sympy import symbols, Matrix
from sympy import lambdify, hessian

import math
import functools as ft                     
import inspect as ins                      
import random as rm                                                        # aleaorio                             

import statistics                            

plt.rcParams["figure.figsize"] = (11,6)
plt.rcParams.update({'font.size': 14})

from matplotlib.pyplot import style                                        # graficos lindos
plt.style.use('seaborn-v0_8')
plt.rc('axes', labelsize=10)
plt.rc('axes', titlesize=10)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.ion()                                                                  

# Funciones 
#-------------------------------
def lineal(x,a,b):
  "función lineal"
  return np.array(x) * a + b

def modulo(x):
    "función módulo"
    if isinstance(x, (int, float)):
        return -x if x < 0 else x
    elif isinstance(x, list):
        return [modulo(val) for val in x]
    else:
        raise ValueError("Unsupported input type. Must be int, float, or list.")

def rad(x):
  "angulos en grados a radianes"
  return x*np.pi/180

def ang(θ):
    "Angulos en radianes a grados"
    return θ*180/np.pi

#-------------------------------

# Normalizar datos                                                         # si se quiere normalizar un valor en vez de una lista usar [valor]
def normalizar_por(valor, lista):
    "normaliza los elementos de una lista por un valor"
    if not isinstance(valor, (int, float)):
        raise TypeError("`valor` must be a number.")
    return [x / valor for x in lista]

# Ordenar datos                                 #(#definir mejor)
def ordenar_por(lista, orden):
  "asocia los valores de la lista a otra a ordenar"
  return [x for _, x in sorted(zip(orden, lista), key=lambda pair: pair[0])]

# Propagación
#-------------------------------

def parametros(f):                                                                                
    """
    Devuelve los parámetros (variables) de f como símbolos de sympy.
    """    
    f_obj=getattr(ins.getmodule(f),f.__name__)
    p = ins.signature(f_obj).parameters
    parametros = [x for x in p]
    ps= [sp.symbols(param) for param in parametros]
    return ps


# Derivadas parciales (evaluadas y no evaluadas)
def derivadas_parciales(f,val=None):                                       # utilizar funciones de sympy (np - sp)
    "calcula las derivadas parciales simbolicas de una función (sympy), puede evaluarse"
    p = parametros(f)
    if val is None: 
        try:
            fs = f(*p)
            return [sp.diff(fs, sym) for sym in p]
        except TypeError as e:
            if "loop of ufunc does not support argument 0 of type" in str(e):
                return "La función utiliza funciones np, se deben utilizar funciones sp"
            else:
                return "No se puede ejecutar la función debido a un TypeError"
        except Exception as e:
            return f"No se puede ejecutar la función: {e}"
    else:
        if not isinstance(val, (list, tuple)):
            val = [val]
        if len(p) != len(val):
            raise ValueError("La cantidad de parámetros y valores no coincide")
        fs = f(*p)
        Df = [sp.diff(fs, sym) for sym in p]
        valores = {param: valor for param, valor in zip(p, val)}
        dfeval= [df.evalf(subs = valores) for df in Df]
        return dfeval

# Propagación
def propagación(f,val,stdval):                                               # utilizar funciones de sympy (np - sp)
    "calcula el error del resultado de una operación a partir de propagación de errores"
    if not isinstance(stdval, (list, tuple)):                                # no utilizar np.arrays
            stdval = [stdval]
    df = derivadas_parciales(f,val)
    return sp.sqrt(sp.Add(*[(derv**2) * (std**2) for derv, std in zip(df, stdval)]))
#-------------------------------

# Máximos
def máximos(x, y, hdt=(0, 1, 0), grafico = False):
    "encuentra los maximos de un set de datos, puede graficarse"
    peaks, _ = find_peaks(y, height=int(hdt[0]), threshold=int(hdt[2]) ,distance=int(hdt[1]))
    xp = [x[pks] for pks in peaks] #ubicación del mxm
    yp = [y[pks] for pks in peaks] #valor del mxm
    
    if grafico:
        plt.plot(x, y, ".b", label='Datos')
        plt.plot(xp, yp, 'o', color='red', label='Picos')
        plt.axhline(hdt[0], color = "red", linewidth = 1, linestyle = "dashed", label="Hd" )
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Picos en los datos')
        plt.legend()
       
    return xp, yp

# Mínimos
def mínimos(x, y, ord=1, grafico = False):
    "encuentra los minimos de un set de datos, puede graficarse"
    y=np.array(y)
    min = argrelmin(y, order=int(ord))[0]
    xm = [x[ms] for ms in min] #ubicación del min
    ym = [y[ms] for ms in min] #valor del min
    
    if grafico:
        plt.plot(x, y, ".b", label='Datos')
        plt.plot(xm, ym, 'o', color='red', label='Mínimos')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Mínimos en los datos')
        plt.legend()

    return xm, ym
#-------------------------------

#Bondad (Chi^2 y p-valor)
def chi2_pvalor(y, yerr, y_mod, parametros):
    "calcula el chi^2 y el p-valor de un ajuste, (devuelve tambien los grados de libertad)"
    y = np.array(y)
    yerr = np.array(yerr)
    def chi2(y_mod, y, yerr):
        return np.sum(((y - y_mod) / yerr)**2)
    grados = len(y) - len(parametros)
    chi_cuadrado = chi2(y_mod, y, yerr)
    p_value = stats.chi2.sf(chi_cuadrado, grados)
    return chi_cuadrado, p_value, grados
# np.array(

#error Chi^2: np.sqrt(2*nu)

#Chi^2 Reducido: normalizar(nu, chi2)
#error np.sqrt(2/nu)

#Coeficiente de Pearson (R^2)
def R2(y,y_mod):
  "calcula el coeficiente de Pearson de un ajuste"
  def residuals(y, ymod):
    return y - y_mod
  ss_res2 = np.sum(residuals(y,y_mod)**2)
  ss_tot2 = np.sum((y-np.mean(y))**2)
  r_squared2 = 1 - (ss_res2 / ss_tot2)
  return r_squared2

#residuos (cuadrados)
#histograma: #bines = np.sqrt(#med)

def residuos(f, pop, x_data, y_data, std, grafico = False, bines = False):
    "calcula los residuos de un ajuste, puede graficarse"
    ymod = np.array(f(x_data, *pop))
    y_data = np.array(y_data)
    res = ((y_data - ymod)**2) 
    resstd = ((y_data - ymod/std)**2)
    if grafico:
        if bines:
            plt.figure()
            plt.title("Histograma de residuos cuadrados")
            plt.hist(res, int(bines))
            plt.figure()
            plt.title("Histograma de residuos cuadrados ponderados")
            plt.hist(resstd, int(bines))
        else:
            plt.figure()
            plt.title("Histograma de residuos cuadrados")
            plt.hist(res, int(np.sqrt(len(y_data))))
            plt.figure()
            plt.title("Histograma de residuos cuadrados ponderados")
            plt.hist(resstd, int(np.sqrt(len(y_data))))
    return res

########################################################################

# Derivadas segundas
#-------------------------------
def derivar_lista(list,p):
  "deriva una lista (de objetos simbolicos) respecto de los parametros p"
  derivadas=[]
  for elemento in list:
    derivadas.append([sp.diff(elemento, x) for x in p])
  return derivadas

def derivadas_parciales_segundas(f_sp, val=None):
  "calcula las derivadas segundas simbolicas de una función (sympy), puede evaluarse"
  p = parametros(f_sp)  
  if val is None: 
    try:
      derivadas_primeras = derivadas_parciales(f_sp)
      derivadas_segundas = derivar_lista(derivadas_primeras, p)
      return derivadas_segundas
    except TypeError as e:
      if "loop of ufunc does not support argument 0 of type" in str(e):
        raise ValueError("Se detectó que la función utiliza funciones de numpy. Por favor, utilice funciones de sympy.") from e
      else:
          raise TypeError("Ha ocurrido un TypeError inesperado: {e}") from e
    except ValueError as e:
        raise ValueError(f"Error de valor en la función: {e}") from e
    except Exception as e:
      raise RuntimeError(f"No se puede ejecutar la función debido a un error inesperado: {e}") from e 
  else:
    if not isinstance(val, (list, tuple)):
      val = [val]
    if len(p) != len(val):
      raise ValueError("La cantidad de parámetros y valores no coincide")
    
    try:
      derivadas_primeras = derivadas_parciales(f_sp)
      derivadas_segundas = derivar_lista(derivadas_primeras, p)
      valores = {param: valor for param, valor in zip(p, val)}
      dfevaltot=[]
      for parcial in derivadas_segundas: 
        dfeval= [df.evalf(subs = valores) for df in parcial]
        dfevaltot.append(dfeval)
      return dfevaltot  
    except Exception as e:
      raise RuntimeError(f"Error durante la evaluación con valores proporcionados: {e}") from e

#-------------------------------

def hessiano(f_sp, param, val = None, total = False):    # param cadena de texto ["a", "b", "c"]  #val debe ser del tamaño de las variables aun etotal True
  "calcula la matriz hessiana simbolica de una función (sympy) respecto de los parametros, con total = True se calcula sobre todas la variables, puede evaluarse"
  p = parametros(f_sp)

  param_syms = [sp.symbols(v) for v in param]
  if not set(param_syms).issubset(set(p)):
    raise ValueError("Los parámetros a optimizar deben ser un subconjunto de las variables de la función.")

  derivadas_segundas = derivadas_parciales_segundas(f_sp, val)

  if total:
      hessian_filtered = derivadas_segundas
  else:                # Filtrar las derivadas segundas correspondientes a los parámetros a optimizar
    indices_param = [p.index(sym) for sym in param_syms]  # Índices de los parámetros relevantes
    hessian_filtered = [
      [derivadas_segundas[i][j] for j in indices_param]  # Filtrar columnas
      for i in indices_param                             # Filtrar filas
    ]

  if val is not None:
    try:
      valores = {p[i]: val[i] for i in range(len(val))}    # Crear un diccionario de sustituciones para los valores
      hessian_filtered = [[entry.evalf(subs=valores) for entry in row] for row in hessian_filtered]
      return np.array(hessian_filtered, dtype=np.float64)  # Convertir la matriz filtrada a formato numpy
    except Exception as e:
      raise TypeError(f"No se pudo evaluar el Hessiano con los valores proporcionados: {e}")
    
  return sp.Matrix(hessian_filtered)                   # Si no hay valores numéricos, devolver la matriz simbólica

#-------------------------------

def jacobiano(fs_sp, param, val=None, total=False):
    "calcula el jacobiano simbolico de una lista de funciones (sympy) respecto de los parametros, con total = True se calcula sobre todas las variables, puede evaluarse"
    if not isinstance(fs_sp, (list, tuple)):
        fs_sp = [fs_sp]  # Convertir a lista si no lo es

    param_syms = [sp.symbols(v) for v in param]
    ps_list = [parametros(f_sp) for f_sp in fs_sp] 

    if not all(set(param_syms).issubset(set(p)) for p in ps_list):
        raise ValueError("Los parámetros a optimizar deben ser un subconjunto de las variables de las funciones.")

    derivadas = [derivadas_parciales(f_sp, val=None) for f_sp in fs_sp]

    if not total:
        # Filtrar columnas correspondientes a los parámetros especificados
        jacobiano_filtered = []
        for f_der, p in zip(derivadas, ps_list):
            indices_param = [p.index(sym) for sym in param_syms]
            jacobiano_filtered.append([f_der[i] for i in indices_param])
    else:
        jacobiano_filtered = derivadas

    if val is not None:
        if not isinstance(val, (list, tuple)):
            val = [val]
        if len(p) != len(val):
            raise ValueError("La cantidad de parámetros y valores no coincide")
        try:
            # Crear un diccionario de sustitución para cada función
            jacobiano_evaluado = []
            for i, f_der in enumerate(jacobiano_filtered):
                valores = {ps_list[i][j]: val[j] for j in range(len(val))}
                jacobiano_evaluado.append([entry.evalf(subs=valores) for entry in f_der])
            return np.array(jacobiano_evaluado, dtype=np.float64)  # Convertir a matriz NumPy
        except Exception as e:
            raise ValueError(f"No se pudo evaluar el Jacobiano con los valores proporcionados: {e}")

    # Si no se proporcionan valores, devolver la matriz simbólica
    return sp.Matrix(jacobiano_filtered)

#-------------------------------
def gradiente(f_sp, val = None):
   "calcula el gradiente simbolico de una función (sympy), puede evaluarse"
   gradiente = np.array(derivadas_parciales(f_sp, val))
   return gradiente

def laplaciano(f_sp, val = None):
  "calcula el laplaciano simbolico de una función (sympy), puede evaluarse"
  p = parametros(f_sp)
  ds = derivadas_parciales_segundas(f_sp)
  terminos = []
  for i in range(len(p)):
      terminos.append(ds[i][i])
  return np.sum(terminos)


########################################################################


#codigo para cargar datos txt/csv de Drive (dueño de la cuenta (MyDrive))
# datos = np.loadtxt("/content/drive/MyDrive/Lab2/" + str(i)  + ".csv" ,delimiter=",", skiprows=1, encoding="latin-1")
# columna1 = datos[:, 0]

import sys

import os
# # Ruta a la carpeta sincronizada de Google Drive en tu sistema de archivos local
# google_drive_folder = '/ruta/a/la/carpeta/sincronizada/de/Google/Drive'
# # Lista todos los archivos en la carpeta sincronizada
# for file_name in os.listdir(google_drive_folder):
#     file_path = os.path.join(google_drive_folder, file_name)
#     if os.path.isfile(file_path):
#         print(f'Archivo encontrado: {file_name}')

#-------------------------------

# para cargar a notebook
# %run 'path_to_config/common_settings.py'

##############################################################



# A corregir y mejorar
##################################################################################
Metodos = ["Nelder-Mead", "Powell", "BFGS", "L-BFGS-B", "CG", "Newton-CG", "TNC", "COBYLA", "SLSQP", "dogleg", "trust-constr", "trust-ncg", "trust-exact", "trust-krylov"] #curvefit (COBYQA)
def Minimizer(f, x_data, y_data, std, parametros_iniciales, metodo = None, opciones = None):          #usar funciones que tomen np.arrays
    "Metodos: Nelder-Mead, Powell, BFGS, L-BFGS-B, CG, Newton-CG, TNC, COBYLA, COBYQA, SLSQP, dogleg, trust-constr, trust-ncg, trust-exact, trust-krylov"
    def error(parametros):
        y_mod = f(x_data, *parametros)
        return np.sum(((y_data - y_mod)/std)**2)

    def jacobiano(parametros):
        epsilon = np.sqrt(np.finfo(float).eps)
        return np.array([(error(parametros + epsilon * np.eye(1, len(parametros), k)[0]) - error(parametros)) / epsilon for k in range(len(parametros))], dtype = float)

    def hessiano(parametros):
        epsilon = np.sqrt(np.finfo(float).eps)
        n = len(parametros)
        hess = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                ei = np.eye(1, n, i)[0] * epsilon
                ej = np.eye(1, n, j)[0] * epsilon
                hess[i, j] = (error(parametros + ei + ej) - error(parametros + ei) - error(parametros + ej) + error(parametros)) / (epsilon ** 2)
        return hess

    jac = jacobiano if metodo in ['Newton-CG', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact'] else None
    hess = hessiano if metodo in ['trust-ncg', 'trust-krylov', 'trust-exact'] else None
    
    resultado = minimize(error, parametros_iniciales, method=metodo, jac=jac, hess=hess, options=opciones)

    return resultado.x
#################################################################################################
