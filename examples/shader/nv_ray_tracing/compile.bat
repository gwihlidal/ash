echo off

glslangValidator -V -o triangle.glsl_rchit.spv triangle.rchit.glsl
glslangValidator -V -o triangle.glsl_rgen.spv triangle.rgen.glsl
glslangValidator -V -o triangle.glsl_rmiss.spv triangle.rmiss.glsl

pause