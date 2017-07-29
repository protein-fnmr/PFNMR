//  PFNMR - Estimate FNMR for proteins (to be updated)
//      Copyright(C) 2016 Jonathan Ellis and Bryan Gantt
//
//  This program is free software : you can redistribute it and / or modify
//      it under the terms of the GNU General Public License as published by
//      the Free Software Foundation, either version 3 of the License.
//
//      This program is distributed in the hope that it will be useful,
//      but WITHOUT ANY WARRANTY; without even the implied warranty of
//      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
//      GNU General Public License for more details.
//
//      You should have received a copy of the GNU General Public License
//      along with this program.If not, see <http://www.gnu.org/licenses/>.

/*
 *  Gauss-Legendre Quadrature is a strange mathematical quirk, but aweseome.
 *  You can (in theory) define any number of points (n) you actually want to
 *  integrate and have it be as accurate as if integrating a 2n + 1 polynomial.
 *  More points is more accurate, but obviously requires more time to
 *  calculate. Adaptation is one way around needing more points, but comes
 *  at a price of potentially being more computationally intensive as the
 *  number of required points doubles each pass. e.g. (potenially bad scenario).
 *
 *  Ten-point adaptive: 10 points first pass  (one 21 degree polynomial)
 *                    + 20 points second pass (two 21 degree polynomials)
 *                    + 40 points third pass  (three 21 degree polynomials)
 *                    = 70 points in total (minimum)
 *
 *  Twenty-point plain: 20 points single pass (one 41 degree poly. MAYBE inaccurate)
 *
 *  This will become a VERY big deal when implementing in array-based GQ as
 *  determining the values of the "function" at the abscissae is already
 *  computationally intensive. Every pass is going to take much more time than
 *  the previous pass.
*/

#include <omp.h>
#include <cmath>
#include <vector>
#include <iostream>

#include "GaussQuadrature.h"

#define max(X, Y) (((X) > (Y)) ? (X) : (Y))

float GaussQuadrature::Intergrate(float begin, float end, float (*func)(const float&))
{
    std::vector<std::vector<float>> GQPoints;
    int count;

    // the values being pushed are doubles converted to floats
    // as floats can have 6-9 decimal places, this should ensure
    // the maximum is used for each possible value at compile time
    switch (mMethod)
    {
    case TWENTY:
        count = 20;
        GQPoints.push_back({ 0.0176140071391521f, -0.9931285991850949f });
        GQPoints.push_back({ 0.0406014298003869f, -0.9639719272779138f });
        GQPoints.push_back({ 0.0626720483341091f, -0.9122344282513259f });
        GQPoints.push_back({ 0.0832767415767048f, -0.8391169718222188f });
        GQPoints.push_back({ 0.1019301198172404f, -0.7463319064601508f });
        GQPoints.push_back({ 0.1181945319615184f, -0.6360536807265150f });
        GQPoints.push_back({ 0.1316886384491766f, -0.5108670019508271f });
        GQPoints.push_back({ 0.1420961093183820f, -0.3737060887154195f });
        GQPoints.push_back({ 0.1491729864726037f, -0.2277858511416451f });
        GQPoints.push_back({ 0.1527533871307258f, -0.0765265211334973f });
        GQPoints.push_back({ 0.1527533871307258f, 0.0765265211334973f });
        GQPoints.push_back({ 0.1491729864726037f, 0.2277858511416451f });
        GQPoints.push_back({ 0.1420961093183820f, 0.3737060887154195f });
        GQPoints.push_back({ 0.1316886384491766f, 0.5108670019508271f });
        GQPoints.push_back({ 0.1181945319615184f, 0.6360536807265150f });
        GQPoints.push_back({ 0.1019301198172404f, 0.7463319064601508f });
        GQPoints.push_back({ 0.0832767415767048f, 0.8391169718222188f });
        GQPoints.push_back({ 0.0626720483341091f, 0.9122344282513259f });
        GQPoints.push_back({ 0.0406014298003869f, 0.9639719272779138f });
        GQPoints.push_back({ 0.0176140071391521f, 0.9931285991850949f });
        break;
    case TEN:
    default:
        count = 10;
        GQPoints.push_back({ 0.0666713443086881f, -0.973906528517171f });
        GQPoints.push_back({ 0.1494513491505800f, -0.865063366688984f });
        GQPoints.push_back({ 0.2190863625159820f, -0.679409568299024f });
        GQPoints.push_back({ 0.2692667193099960f, -0.433395394129247f });
        GQPoints.push_back({ 0.2955242247147520f, -0.148874338981631f });
        GQPoints.push_back({ 0.2955242247147520f, 0.148874338981631f });
        GQPoints.push_back({ 0.2692667193099960f, 0.433395394129247f });
        GQPoints.push_back({ 0.2190863625159820f, 0.679409568299024f });
        GQPoints.push_back({ 0.1494513491505800f, 0.865063366688984f });
        GQPoints.push_back({ 0.0666713443086881f, 0.973906528517171f });
    }

    float solution = 0.0, prevSolution = 0.0;
    int i, divisions = 1;
    float sum = 0.0, newA, newB, halfBpa, halfBma, size, x;

    // for now we should just assume the maxium number of logical CPUs + 1
    auto maxThreads = omp_get_num_procs() + 1;

    while ((bAdaptive && (abs(prevSolution - solution) > fEps)) || divisions == 1)
    {
        prevSolution = solution;
        solution = 0.0f;

        omp_set_num_threads(max(divisions, maxThreads));

        size = (end - begin) / divisions;

        #pragma omp parallel shared(size, divisions, count, solution, prevSolution) private(i, x, halfBma, halfBpa, newA, newB)
        #pragma omp for reduction (+:sum)
        for (i = 0; i < divisions; ++i)
        {
            newA = begin + size * i;
            newB = newA + size;
            halfBpa = (newB + newA) / 2;
            halfBma = (newB - newA) / 2;

            for (int j = 0; j < count; ++j)
            {
                x = halfBma * GQPoints[j][1] + halfBpa;
                sum += GQPoints[j][0] * func(x);
            }

            #pragma omp critical
            solution += (size / 2) * sum;
        }
        
        if (bAdaptive)
            divisions *= 2;
        else
            break;
    }

    return solution;
}
