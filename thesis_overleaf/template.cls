\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{template}[2026/11/05 Compatible BTP report class]
% A supplied compatibility class, NOT an official IIT Kharagpur class.
\LoadClass[12pt,a4paper,oneside]{report}
\RequirePackage[left=32mm,right=25mm,top=27mm,bottom=27mm,headheight=16pt]{geometry}
\RequirePackage{setspace}
\RequirePackage{fancyhdr}
\RequirePackage{xcolor}
\RequirePackage{titlesec}
\onehalfspacing
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small\nouppercase{\leftmark}}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\renewcommand{\chaptermark}[1]{\markboth{\thechapter.\ #1}{}}
\titleformat{\chapter}[display]{\normalfont\bfseries\LARGE\raggedright}{\chaptername\ \thechapter}{10pt}{\Huge}
\titlespacing*{\chapter}{0pt}{0pt}{22pt}
\newcommand{\degreetitle}[1]{\def\@degreetitle{#1}}
\newcommand{\rpttype}[1]{\def\@rpttype{#1}}
\newcommand{\principaladviser}[1]{\def\@principaladviser{#1}}
\newcommand{\studentnumber}[1]{\def\@studentnumber{#1}}
\newcommand{\department}[1]{\def\@department{#1}}
\newcommand{\institute}[1]{\def\@institute{#1}}
\renewcommand{\maketitle}{\begin{titlepage}\centering
\vspace*{8mm}
{\LARGE\bfseries \@title\par}\vspace{15mm}
{\Large \@rpttype\ Report\par}\vspace{5mm}
{\large \@degreetitle\ Programme\par}\vspace{15mm}
{\large Submitted by\par}\vspace{3mm}
{\Large\bfseries \@author\par}\vspace{2mm}
{\large Roll Number: \@studentnumber\par}\vspace{12mm}
{\large Supervisor\par}\vspace{3mm}
{\Large \@principaladviser\par}\vfill
{\large Department of \@department\par}\vspace{3mm}
{\Large\bfseries \@institute\par}\vspace{6mm}
{\large \@date\par}\vspace{8mm}
\end{titlepage}}
