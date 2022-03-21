rm -rf doc/*
javadoc -classpath lib/jblas-1.2.5.jar:. -d doc $(find minet -name "*.java")
