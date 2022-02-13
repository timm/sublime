.PHONY: help tests ho bye mds

help:
	@printf "\nmake [OPTIONS]\n\nOPTIONS:\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| sort \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%10s :\033[0m %s\n", $$1, $$2}'

ok:
	@mkdir -p ../docs/pdf

hi: ## start work (update all files)
	git add *;git commit -am save;git push;git status

bye: mds ## stop work (save all files)
	git add *;git commit -am save;git push;git status

pdoc: ok ## generate docs (python)
	pdoc --logo "https://raw.githubusercontent.com/timm/sublime/main/etc/img/lime.png" \
       -o ../docs sublime.py

pdfs: ../docs/pdf/sublime.pdf ## generate docs (pdf)

../docs/pdf/%.pdf : %.py ok
	@a2ps -q -BjR --line-numbers=1               \
           --borders=no --pro=color --columns 2 \
           --right-footer="" --left-footer=""    \
           --footer="page %p."                    \
            --pretty-print=../etc/python.ssh       \
           -M letter -o $@.ps $<
	@ps2pdf $@.ps $@; rm $@.ps ; git add $@


