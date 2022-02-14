.PHONY: help tests ho bye mds

help:
	@printf "\n[SETTINGS] make [OPTIONS]\n\nOPTIONS:\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| sort \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%10s :\033[0m %s\n", $$1, $$2}'

demo: ## run demo $t; e.g. t=all make demo 
	@python3 sublime.py -t $t; printf "\n\nexit status: $$?\n\n"

all: ok pdoc pdfs bye ## run all (i.e. make ok pdoc pdfs bye)
	echo $(MAKEFILE)

ok:
	@mkdir -p docs/pdf

hi: ## start work (update all files)
	git pull

bye: mds ## stop work (save all files)
	git add *;git commit -am save;git push;git status

pdoc: ok $(MAKEFILE) sublime.py ## generate docs (python)
	pdoc --logo "https://raw.githubusercontent.com/timm/sublime/main/etc/img/lime.png" \
       -o docs sublime.py

pdfs: docs/pdf/sublime.pdf ## generate docs (pdf)

docs/pdf/%.pdf : %.py ok
	@a2ps -q -BjR --line-numbers=1               \
           --borders=no --pro=color --columns 2 \
           --right-footer="" --left-footer=""    \
           --footer="page %p."                    \
            --pretty-print=etc/python.ssh       \
           -M letter -o $@.ps $<
	@ps2pdf $@.ps $@; rm $@.ps ; git add $@


