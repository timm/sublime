.PHONY: help all bye demo hi ok pdoc pdfs

help: ## show help
	@printf "\n[SETTINGS] make [OPTIONS]\n\nOPTIONS:\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| sort \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%10s :\033[0m %s\n", $$1, $$2}'

all: ok pep8s pdoc pdfs bye ## run all (i.e. make ok pdoc pdfs bye)

bye:  ## stop work (save all files)
	git add *;git commit -am save;git push;git status

demo: ## run demo $t; e.g. t=all make demo 
	@python3 sublime.py -t $t; printf "\n\nexit status: $$?\n\n"

hi: ## start work (update all files)
	git pull

pep8s: sublime-pep8.py 
pdoc: docs/sublime.html   ## generate docs (html)
pdfs: docs/pdf/sublime.pdf docs/pdf/sublime-pep8.pdf ## generate docs (pdf)

#----------------------------------------------------
ok:
	@mkdir -p docs/pdf

%-pep8.py: %.py
	autopep8 --indent-size=2 $< > $@

docs/%.html : %.py  
	pdoc --logo "https://raw.githubusercontent.com/timm/sublime/main/etc/img/lime.png" \
       -o docs sublime.py

docs/pdf/%.pdf : %.py 
	@a2ps -q -BjR --line-numbers=1               \
           --borders=no --pro=color --columns 2 \
           --right-footer="" --left-footer=""    \
           --footer="page %p."                    \
            --pretty-print=etc/python.ssh       \
           -M letter -o $@.ps $<
	@ps2pdf $@.ps $@; rm $@.ps ; git add $@


