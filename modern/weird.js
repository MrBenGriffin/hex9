// https://observablehq.com/@jrus/conformal-octahedron@2385
function _Title(md){return(
md`
# Conformal octahedron`
)}

function _2(DOM,width,height,d3,projection,land50,sphere,graticule2,graticule,drag,land110)
{
  const context = DOM.context2d(width, height);
  context.canvas.classList.add('draggable');
  const path = d3.geoPath(projection, context);
  let land = land50;

  function render(x, y) {
    context.fillStyle = "#ffffff"; //"#f7f8fb";
    context.fillRect(0, 0, width, height);

    context.fillStyle = "#fffffa";
    context.beginPath(), path(sphere), context.fill();



    context.strokeStyle = "#cdf";
    context.lineWidth = 0.75;
    context.beginPath(), path(graticule2), context.stroke();

    context.strokeStyle = "#7be";
    context.lineWidth = 0.85;
    context.beginPath(), path(graticule), context.stroke();

    context.fillStyle = 'rgba(10, 40, 24, .88)'
    context.beginPath(), path(land), context.fill();

    context.lineWidth = 1.0;
    context.strokeStyle = "#999";
    context.beginPath(), path(sphere), context.stroke();

  }

  return d3.select(context.canvas)
   .call(drag(projection)
       .on("start.render", () => {land = land110; render();})
       .on("drag.render", () => render())
       .on("end.render", () => {land = land50; render();}))
    .call(() => render())
    .node();
}


function _3(md){return(
md`

Make sure you try dragging the map above.

* * *

Oscar Sherman Adams, a senior geodetic computer for the U.S. Coast and Geodetic Survey, first constructed a conformal projection from the sphere to the octahedron circa 1925, under contract from Bernard Joseph Stanislaus Cahill. For historical details see [Gene Keyes’s website](http://www.genekeyes.com/CAHILL-VARIANTS/Cahill-Conformal.html).

Decades later, Laurence Patrick Lee, chief geodetic computer for the the New Zealand Lands and Survey Department, described this and similar projections in a 1976 monograph, [*Conformal Projections Based on Elliptic Functions*](https://archive.org/details/conformalproject0000leel).

* * *

Thanks to [Philippe Rivière](../@fil) for significant help understanding [d3-geo-polygon](https://github.com/d3/d3-geo-polygon).

* * *

`
)}

function _4(md){return(
md`
Here’s a single octant, plotted with a 1° graticule near the equator and coarser longitudes toward the poles. Again, please try dragging the map:
`
)}

function _5(width,butterfly,octoformal_octant,sphere,d3,DOM,land50,drag,land110)
{
  const height = Math.ceil(width * Math.sqrt(3)/2)

  const projection = butterfly(octoformal_octant, [-1])
      .center([45, 50])
      .angle(-90)
      .rotate([20,0,0])
      .fitExtent([[1 , 1], [width - 1, height - 1]], sphere)
      .precision(0.05);

  const graticule = d3.geoGraticule()
    .extentMinor([[-180, -90], [180, 90]])
    .stepMinor([10, 10])();

  const graticule2 = d3.geoGraticule()
    .extentMajor([[-180, -90], [180, 90]])
    .extentMinor([[-180, -90], [180, 90]])
    .stepMajor([90, 90])
    .stepMinor([90, 1])()

  const graticule3 = d3.geoGraticule()
    .extentMajor([[-180, -85 -1e-5], [180, 85 + 1e-5]])
    .extentMinor([[-180, -60 -1e-5], [180, 60 + 1e-5]])
    .stepMajor([2, 90])
    .stepMinor([1, 90])()

  const context = DOM.context2d(width, height);
  context.canvas.classList.add('draggable');
  const path = d3.geoPath(projection, context);
  let land = land50;

  function render(x, y) {
    context.fillStyle = "#ffffff"; //"#f7f8fb";
    context.fillRect(0, 0, width, height);

    context.fillStyle = "#fffffa";
    context.beginPath(), path(sphere), context.fill();

    context.strokeStyle = "#cfdfff";
    context.lineWidth = 0.75;
    context.beginPath(), path(graticule2), context.stroke();

    context.strokeStyle = "#cfdfff";
    context.lineWidth = 0.75;
    context.beginPath(), path(graticule3), context.stroke();

    context.strokeStyle = "#9ce";
    context.lineWidth = 0.85;
    context.beginPath(), path(graticule), context.stroke();

    context.fillStyle = 'rgba(10, 40, 24, .80)'
    context.beginPath(), path(land), context.fill();

    context.lineWidth = 0.85;
    context.strokeStyle = "#999";
    context.beginPath(), path(sphere), context.stroke();
  }

  return d3.select(context.canvas)
   .call(drag(projection)
       .on("start.render", () => {land = land110; render();})
       .on("drag.render", () => render())
       .on("end.render", () => {land = land50; render();}))
    .call(() => render())
    .node();



}


function _6(md){return(
md`
@Fil also made a spinoff notebook including Cahill’s “M” and “Zigzag” arrangements, in addition to the butterfly map. We can reproduce Cahill’s maps with their characteristic 22°30′ (1/16 [turns][]) longitude rotation.

  [turns]: https://en.wikipedia.org/wiki/Turn_(angle)

[<img src=http://genekeyes.com/CAHILL-VARIANTS/Cahill-Conformal-M.jpg>](http://www.genekeyes.com/CAHILL-VARIANTS/Cahill-Conformal.html)
`
)}

function _7(width,butterfly,octoformal_octant,sphere,DOM,d3,land50,graticule2,graticule,drag,land110)
{
  const height = Math.ceil(width / Math.sqrt(3) * 3/4)

  const projection = butterfly(octoformal_octant, [-1, 3, 0, 7, 0, 1, 2, 6])
      .center([180, -20])
      .angle(-120)
      .rotate([22.5,0,0])
      .fitExtent([[1 , 1], [width - 1, height - 1]], sphere)
      .precision(0.05);

  const context = DOM.context2d(width, height);
  context.canvas.classList.add('draggable');
  const path = d3.geoPath(projection, context);
  let land = land50;

  function render(x, y) {
    context.fillStyle = "#ffffff"; //"#f7f8fb";
    context.fillRect(0, 0, width, height);

    context.fillStyle = "#fffffa";
    context.beginPath(), path(sphere), context.fill();

    context.strokeStyle = "#cdf";
    context.lineWidth = 0.75;
    context.beginPath(), path(graticule2), context.stroke();

    context.strokeStyle = "#7be";
    context.lineWidth = 0.85;
    context.beginPath(), path(graticule), context.stroke();

    context.fillStyle = 'rgba(10, 40, 24, .88)'
    context.beginPath(), path(land), context.fill();

    context.lineWidth = 1.0;
    context.strokeStyle = "#999";
    context.beginPath(), path(sphere), context.stroke();

  }

  return d3.select(context.canvas)
   .call(drag(projection)
       .on("start.render", () => {land = land110; render();})
       .on("drag.render", () => render())
       .on("end.render", () => {land = land50; render();}))
    .call(() => render())
    .node();



}


function _8(md){return(
md`
[<img src=https://www.genekeyes.com/Cahill-To-End-World-Maps/fig.2b+.jpg>](http://www.genekeyes.com/CAHILL-VARIANTS/Cahill-Conformal.html)
`
)}

function _9(width,butterfly,octoformal_octant,sphere,DOM,d3,land50,graticule2,graticule,drag,land110)
{
  const height = Math.ceil(width / Math.sqrt(3))

  const projection = butterfly(octoformal_octant, [-1, 0, 0, 2, 5, 7, 4, 3])
      .center([135, 0])
      .angle(-120)
      .rotate([112.5,0,0])
      .fitExtent([[1 , 1], [width - 1, height - 1]], sphere)
      .precision(0.05);

  const context = DOM.context2d(width, height);
  context.canvas.classList.add('draggable');
  const path = d3.geoPath(projection, context);
  let land = land50;

  function render(x, y) {
    context.fillStyle = "#ffffff"; //"#f7f8fb";
    context.fillRect(0, 0, width, height);

    context.fillStyle = "#fffffa";
    context.beginPath(), path(sphere), context.fill();

    context.strokeStyle = "#cdf";
    context.lineWidth = 0.75;
    context.beginPath(), path(graticule2), context.stroke();

    context.strokeStyle = "#7be";
    context.lineWidth = 0.85;
    context.beginPath(), path(graticule), context.stroke();

    context.fillStyle = 'rgba(10, 40, 24, .88)'
    context.beginPath(), path(land), context.fill();

    context.lineWidth = 1.0;
    context.strokeStyle = "#999";
    context.beginPath(), path(sphere), context.stroke();

  }

  return d3.select(context.canvas)
   .call(drag(projection)
       .on("start.render", () => {land = land110; render();})
       .on("drag.render", () => render())
       .on("end.render", () => {land = land50; render();}))
    .call(() => render())
    .node();



}


function _10(md){return(
md`---`
)}

function _projection(butterfly,octoformal_octant,width,height,sphere)
{
  const {cos, sin, sign} = Math;
  const RADIANS = Math.PI / 180;

  return butterfly(octoformal_octant)
    .angle(-120)
    .rotate([20,0,0])
    .fitExtent([[1 , 1], [width - 1, height - 1]], sphere)
    .precision(0.05);
}


function _octahedron(){return(
[0,1,2,3,4,5,6,7].map((i) => {
  const negz = i >> 2;
  const negx = (i >> 1) & 1;
  const negy = i & 1
  const zvertex = [[0, 90], [0, -90]][negz];
  const xvertex = [[0, 0],  [180, 0]][negx];
  const yvertex = [[90, 0], [-90, 0]][negy];
  if (negz ^ negx ^ negy)
    return [zvertex, xvertex, yvertex];
  return [zvertex, yvertex, xvertex]
})
)}

function _octahedronExplicit()
{
  var vertices = [
    [0, 90], [0, -90], // z+, z-
    [0, 0],  [180, 0], // x+, x-
    [90, 0], [-90, 0]  // y+, y-
  ];


  return [     // z x y
    [0, 4, 2], // + + +
    [0, 2, 5], // + + -
    [0, 3, 4], // + - +
    [0, 5, 3], // + - -
    [1, 2, 4], // - + +
    [1, 5, 2], // - + -
    [1, 4, 3], // - - +
    [1, 3, 5]  // - - -
  ].map((face) => face.map((i) => vertices[i]));
}


function _butterfly(octahedron,d3){return(
function(faceProjection, layout) {
  const TWO_OVER_PI = 2/Math.PI;
  const faces = octahedron.map((face) => ({
    'face': face,
    'project': faceProjection(face)
  }));

  layout = layout || [-1, 0, 0, 2, 0, 1, 2, 3];

  layout.forEach((d, i) => {
    var node = faces[d];
    node && (node.children || (node.children = [])).push(faces[i]);
  });

  const pickFace = function(longitude, latitude) {
    longitude *= TWO_OVER_PI;
    const zneg = latitude < 0,
          yneg = longitude < 0,
          xneg = longitude * longitude > 1;
    return faces[(zneg << 2) + (xneg << 1) + yneg];
  }

  return d3.geoPolyhedral(faces[0], pickFace)
    .angle(-30)
    .scale(100)
    .center([90, 45]);
}
)}

function _octoformal_octant(d3,cartesian_octoformal){return(
function octoformal_octant(face) {
  const {cos, sin, sign, atan2, hypot} = Math;
  const RADIANS = Math.PI / 180;

  let smz, smx, smy;
  { let [mlon, mlat] = d3.geoCentroid({type: "MultiPoint", coordinates: face});
    mlon *= RADIANS; mlat *= RADIANS;
    smz = sign(sin(mlat)),
    smx = sign(cos(mlat)*cos(mlon)),
    smy = sign(cos(mlat)*sin(mlon)); }

  const project_octant = function (lon, lat) {
    let c = cos(lat),
        z = sin(lat), x = c*cos(lon), y = c*sin(lon);

    // truncate xyz coordinates falling outside this octant
    if (smz*z <= 0) z = smz * 0;
    if (smx*x <= 0) x = smx * 0;
    if (smy*y <= 0) y = smy * 0;
    return cartesian_octoformal(z, x, y);
  };

  project_octant.invert = function (x, y) {
    y *= smz * smx * smy; // reflect alternate octants
    const w = cartesian_octoformal.invert(x, y);
    let z = w[0]; x = w[1]; y = w[2]; w.pop();

    // reflect into appropriate output octant
    z *= smz; x *= smx; y *= smy;
    w[0] = atan2(y, x);
    w[1] = atan2(z, hypot(x, y));
    return w;
  }

  return d3.geoProjection(project_octant);
}
)}

function _cartesian_octoformal(omap,oimap)
{
  const {sqrt, sign, round, cbrt, atan2, cos, sin} = Math;
  const TWOTHIRDS = 2/3;
  const SQRT3 = sqrt(3);

  const cartesian_octoformal = function cartesian_octoformal(z, x, y) {

    // ********** DOMAIN REDUCTION

    const sz = sign(z) || sign(1/z); // -0 -> -1; +0 -> +1
    const sx = sign(x) || sign(1/x);
    const sy = sign(y) || sign(1/y);

    // reflect into positive octant
    z *= sz; x *= sx; y *= sy;

    // permute axes so that z >= x >= y
    const ygtx = y > x;
    if (ygtx) { const t = x; x = y; y = t; }
    const ygtz = y > z;
    if (ygtz) { const t = z; z = y; y = t; }
    const xgtz = x >= z;
    if (xgtz) { const t = z; z = x; x = t; }

    // stereographic projection
    z = 1 / (1 + z); x *= z; y *= z;

    // ********** MAP FUNDAMENTAL PIE SLICE

    const w = omap(x, y); // apply rational approximation
    x = w[0]; y = w[1];

    // x + iy = (x + iy)^(2/3)
    { const r = cbrt(x*x + y*y);
      const angle = atan2(y, x) * TWOTHIRDS;
      x = r * cos(angle);
      y = r * sin(angle);
    }
    x -= 1; // put center of the triangle at the origin

    // ********** REFLECT ACCORDING TO EARLIER DOMAIN REDUCTIONS

    { const rot = SQRT3 * ((xgtz & ygtz) - (xgtz ^ ygtz));
      const xrt3 = x * rot;
      const yrt3 = - y * rot;
      x += (xgtz | ygtz) * 0.5 * (yrt3 - 3 * x); // rotate
      y += (xgtz | ygtz) * 0.5 * (xrt3 - 3 * y);
      y *= (1 - 2 * (ygtx ^ ygtz ^ xgtz)); // flip y
    }

    y *= sz * sy * sx; // reflect odd octants

    w[0] = x; w[1] = y; // re-use array z
    return w;
  };

  cartesian_octoformal.invert = function cartesian_octoformal(x, y) {

    // ********** DOMAIN REDUCTION

    const xgtz = (x * SQRT3 > y) | 0,
          ygtz = (x * SQRT3 > -y) | 0,
          ygtx = (y > 0) | 0;
    { const rot = SQRT3 * ((ygtz & ygtx) - (xgtz & (ygtx ^ 1))),
            xrt3 = x * rot,
            yrt3 = - y * rot;
      x += (xgtz | ygtz) * 0.5 * (yrt3 - 3 * x); // rotate
      y += (xgtz | ygtz) * 0.5 * (xrt3 - 3 * y);
      y *= (1 - 2 * (xgtz ^ ygtz ^ ygtx)); // flip y
    }

    x += 1; // move singularity to origin

    // ********** MAP FUNDAMENTAL TRIANGLE

    // x + iy = (x + iy)^(3/2), assuming y started <= 0
    { const t = x, r = sqrt(x*x + y*y),
            x2 = sqrt(0.5 * (r + x)), // x2 + i*y2 = sqrt(x + iy)
            y2 = -sqrt(0.5 * (r - x));
      x = x * x2 - y * y2;
      y = t * y2 + y * x2;
    }

    let w = oimap(x, y); // apply rational approximation
    x = w[0]; y = w[1];

    // inverse stereographic projection
    let z;
    { const d = 1 / (0.5 + 0.5*(x*x + y*y));
      z = (0.5 - 0.5*(x*x + y*y)) * d;
      x *= d;
      y *= d;
    }

    // ********** PERMUTE AXES ACCORDING TO EARLIER DOMAIN REDUCTIONS

    if (xgtz ^ ygtz ^ ygtx) { const t = x; x = y; y = t; }

    if (ygtz & ygtx) { const t = z; z = x; x = y; y = t; }
    else if (xgtz & (ygtx ^ 1)) { const t = z; z = y; y = x; x = t; }

    return [z, x, y];
  }

  return cartesian_octoformal;
}


function _omap(reval)
{
  const nodes = new Float64Array([
     4.100916630039677e-01,  1.078957919786730e-01,
     1.446210672655586e-01,  1.446210672655586e-01,
     1.234872078366100e-01,  0.000000000000000e+00,
     3.880110914840966e-01,  2.709708654396763e-01,
     3.682018550462841e-02,  3.682018550462841e-02,
     3.216272607284061e-01,  3.216272607284061e-01,
     3.213743061551601e-01,  0.000000000000000e+00,
     2.419471033672143e-01,  2.419471033672143e-01,
     3.662644231234737e-01,  3.651322036017664e-01,
     2.171645054002400e-02,  0.000000000000000e+00,
     4.141488339214348e-01,  1.353054022004537e-02
  ]);
  const values = new Float64Array([
     7.128058932401936e-01, -4.139440993595627e-01,
     3.970467265727428e-01,  0.000000000000000e+00,
     1.695366102332019e-01, -1.695366102332019e-01,
     9.026063887759997e-01, -1.582075029409894e-01,
     1.010997427064757e-01,  0.000000000000000e+00,
     8.804567105023327e-01,  0.000000000000000e+00,
     4.415476272745202e-01, -4.415476272745202e-01,
     6.636841839031136e-01,  0.000000000000000e+00,
     9.991239427727002e-01, -1.516489767790573e-03,
     2.981420000798688e-02, -2.981420000798688e-02,
     5.885482041415332e-01, -5.509991640846738e-01
  ]);
  const weights = new Float64Array([
    -1.604010799036173e-01,  0.000000000000000e+00,
     4.406932638137102e-01, -3.590142358058550e-01,
    -2.682474870656388e-01,  3.233841263826757e-01,
    -3.456311610501484e-02,  7.816337352508099e-02,
     4.060412268290973e-01, -1.175759772628968e-01,
    -1.234674387645387e-01,  1.535188098943858e-02,
     1.589851191228840e-02,  2.164613645838719e-01,
    -1.355901340191331e-01, -3.974410641906075e-01,
     9.763735447473679e-03,  5.433180299326212e-03,
    -9.163015921366578e-02,  1.886629075998683e-01,
    -5.849732155961733e-02,  4.657443976479696e-02
  ]);
  return reval(nodes, values, weights);
}


function _oimap(reval)
{
  const nodes = new Float64Array([
     7.121620036554932e-01, -4.147098356997996e-01,
     4.026760197464420e-01, -0.000000000000000e+00,
     1.676565322021649e-01, -1.676565322021649e-01,
     1.111509242644564e-01, -0.000000000000000e+00,
     8.173136328251600e-01, -0.000000000000000e+00,
     9.219502964738188e-01, -1.283941485107439e-01,
     4.152344775110253e-01, -4.152344775110253e-01,
     5.865914838499277e-01, -5.529970199924502e-01,
     9.936882763431703e-01, -1.088635423721026e-02
  ]);
  const values = new Float64Array([
     4.101306536000851e-01,  1.073850072281838e-01,
     1.466725531212872e-01,  1.466725531212872e-01,
     1.221178836352449e-01,  0.000000000000000e+00,
     4.048080174741815e-02,  4.048080174741815e-02,
     2.983300885674245e-01,  2.983300885674245e-01,
     3.843319729813929e-01,  2.891798550754255e-01,
     3.022730272522167e-01,  0.000000000000000e+00,
     4.141617509407632e-01,  1.210546059242303e-02,
     3.677293652515539e-01,  3.596058723499682e-01
  ]);
  const weights = new Float64Array([
     2.454639436696862e-01,  0.000000000000000e+00,
    -2.062366592193658e-01,  4.314467410113785e-01,
    -1.308918044184494e-01, -2.757937226001616e-01,
    -1.154451887326551e-01,  5.739532038113065e-02,
    -1.327290526250629e-01,  4.905901500513447e-01,
     2.218695459619175e-01, -3.108343262633364e-01,
    -1.051671671849686e-01, -3.811564818299064e-01,
     9.003881303601896e-02, -9.208213761650617e-02,
     1.330976078818440e-01,  8.043443133079413e-02
  ]);
  return reval(nodes, values, weights);
}


function _reval(){return(
function reval(nodes, values, weights) {
  const n = nodes.length;

  return function reval(x, y) {

    // running total for numerator & denominator
    let px = 0, py = 0, qx = 0, qy = 0;

    for (let j = 0; j < n; j += 2) {
      // q += w_j / (z - z_j)
      // p += f_j * w_j / (z - z_j)
      const xj = x - nodes[j], yj = nodes[j+1] - y,
            wxj = weights[j], wyj = weights[j+1],
            fxj = values[j], fyj = values[j+1],
            dj = 1 / (xj*xj + yj*yj),
            qxj = dj * (wxj*xj - wyj*yj),
            qyj = dj * (wxj*yj + wyj*xj);
      px += fxj * qxj - fyj * qyj;
      py += fxj * qyj + fyj * qxj;
      qx += qxj, qy += qyj;
    }

    let d = 1 / (qx * qx + qy * qy),
        fx = (px * qx + py * qy) * d,
        fy = (py * qx - px * qy) * d;

    // Edge case where x + iy is one of the nodes; directly use value
    if (fx + fy !== fx + fy) { // true if x or y is NaN
      for (let j = 0; j < n; j += 2) {
        const xj = x - nodes[j], yj = nodes[j+1] - y;
        if ((xj === 0) & (yj === 0)) {
          fx = values[j]; fy = values[j+1];
          break;
        }
      }
    }

    const z = new Float64Array(2); z[0] = fx; z[1] = fy;
    return z;
  };

}
)}

function _20(md){return(
md`
* * *
Below is Matlab code to generate the nodes, values, and weights used in this rational approximation. (Relies on Toby Driscoll’s [SC Toolbox](http://www.math.udel.edu/~driscoll/SC/) and Nick Trefethen & al.’s [Chebfun](http://www.chebfun.org), specifically the [AAA](https://github.com/chebfun/chebfun/blob/master/aaa.m) tool based on Nakatsukasa, Sète, & Trefethen (2018) [“The AAA Algorithm for Rational Approximation”](http://people.maths.ox.ac.uk/trefethen/AAAfinal.pdf). Also cf. Gopal & Trefethen (2019) [“Representation of Conformal Maps by Rational Functions”](https://arxiv.org/pdf/1804.08127.pdf).)
* * *

\`\`\`matlab
% rational quadratic circle-arc interpolant, a function f
% such that f(0) = a, f(1/2) = m, f(1) = b
rerp = @(a, m, b) ...
  (@(z) ((b-m).*a.*(1-z) + (m-a).*b.*z) ...
     ./ ((b-m).*(1-z) + (m-a).*z));

lerp = @(a, b) ...
  (@(z) a.*(1-z) + b.*z);

% midpoint between two points on the sphere, represented stereographically
midpoint = @(w, z) ...
 (z + w + z.*w.*conj(z + w)) ./ ...
 ( abs(1 + z.*conj(w)) .* sqrt((1 + z.*conj(z)).* (1 + w.*conj(w))) + ...
   1 - z .* conj(z) .* w .* conj(w));

spoke3 = rerp(0, ...
  midpoint(0, sqrt(2) / (1 + sqrt(3))), ...
  sqrt(2) / (1 + sqrt(3)));

spoke2 = rerp(0, ...
  midpoint(0, 1 / (1 + sqrt(2))),  ...
  1 / (1 + sqrt(2)));

arc23 = rerp( ...
  1 / (1 + sqrt(2)), ...
  midpoint(1 / (1 + sqrt(2)), (1 + i) / (1 + sqrt(3))), ...
  (1 + i) / (1 + sqrt(3)));

tri = hplmap( ...
  [-1 1 Inf], ...
  [1/3 1/2 1/6], ...
  scmapopt('Tolerance', 1e-16));
trimap = @(z) (tri(z) - tri(Inf)) ./ (tri(7) - tri(Inf));

sector_to_halfplane = @(z) -cosh(4 * log(z));

nf = 64;
fz = chebpts(2*nf + 1);
fz = fz(nf+2:end-1); % take points from 0 .. 1
fz = fz .* (sqrt(2) / (1 + sqrt(3)));
fz = - sector_to_halfplane(fz);
fz = real(trimap(fz)) .^ (3/2);
fz = [-1; -flipud(fz); 0; fz; 1];
f = simplify(chebfun(fz, [ ...
  -sqrt(2) / (1 + sqrt(3)), ...
  sqrt(2) / (1 + sqrt(3))]));

ng = 64;
gz = chebpts(2*ng + 1);
gz = gz(ng+2:end-1); % take points from 0 .. 1
gz = gz .* (1 / (1 + sqrt(2)));
gz = sector_to_halfplane(gz);
gz = real(trimap(gz) .^ (3/2));
gz = [-3^(3/4)/4; -flipud(gz); 0; gz; 3^(3/4)/4];
g = simplify(chebfun(gz, [ ...
  -1/(1 + sqrt(2)), ...
  1/(1 + sqrt(2))]));

nh = 64;
hz = chebpts(2*nh + 1, [0 1]);
hz = hz(2:end-1);
hz = arc23(hz);
hz = sector_to_halfplane(hz);
hz = trimap(hz);
hz = [((3 - sqrt(3)*1i)/4); hz; 1];
h = simplify(chebfun(4 * real(hz) - 3, [0 1]));


% points to use per side
n = 1024;

p = chebpts(n+1, [0 1]); p = p(2:end-1);

% input points around the boundary of the half-quadrant,
% spaced to be packed closer near the corner with the
% square root singularity
zin = [  ...
  0; ...
  spoke2(p.^2); ...
  1/(1+sqrt(2)); ...
  arc23(p); ...
  (1+1i)/(1+sqrt(3)); ...
  ((1+1i)/sqrt(2)) * flipud(spoke3(p.^2)) ...
  ];

shortside = lerp(3/4 - 1i*sqrt(3)/4, 1);

zout = [ ...
  0; ...
  (1-1i) * g(spoke2(p.^2)); ...
  (1-1i) * 3^(3/4)/4; ...
  shortside(h(p)) .^ (3/2); ...
  1; ...
  flipud(f(spoke3(p.^2))) ...
  ];

format longE;

[r, rpol, ~, ~, rz, rf, rw] = aaa(zout(1:2:end), zin(1:2:end), 'tol', 3e-15)
[s, spol, ~, ~, sz, sf, sw] = aaa(zin, zout, 'tol', 4e-15)

% these approximations are pretty good!
norm(zout - r(zin), inf)
norm(s(zout) - zin, inf)

figure; daspect([1 1 1]); hold on; plot(zin); plot(rz, 'ko'); plot(rpol, 'ro'); 
figure; daspect([1 1 1]); hold on; plot(zout); plot(sz, 'ko'); plot(spol, 'ro');
\`\`\`

`
)}

function _21(md){return(
md`---`
)}

function _height(width){return(
Math.ceil(width / Math.sqrt(3))
)}

function _sphere(){return(
{ type: "Sphere" }
)}

function _graticule(d3){return(
d3.geoGraticule()
  .extentMinor([[-180, -85], [180, 85]])
  .stepMinor([30, 30])()
)}

function _graticule2(d3){return(
d3.geoGraticule()
  .extentMajor([[-180, -85 -1e-5], [180, 85 + 1e-5]])
  .extentMinor([[-180, -80], [180, 80]])
  .stepMajor([10, 5])
  .stepMinor([5, 5])()
)}

function _land50(topojson,world50){return(
topojson.feature(world50, world50.objects.land)
)}

function _land110(topojson,world110){return(
topojson.feature(world110, world110.objects.land)
)}

function _world50(){return(
fetch("https://cdn.jsdelivr.net/npm/world-atlas@2/land-50m.json").then(response => response.json())
)}

function _world110(){return(
fetch("https://cdn.jsdelivr.net/npm/world-atlas@2/land-110m.json").then(response => response.json())
)}

function _topojson(require){return(
require("topojson-client@3")
)}

function _d3(require){return(
require("d3-geo@1", "d3-selection@1", "d3-drag@1", "d3-geo-projection@2", "d3-geo-polygon@1")
)}

function _drag(versor,d3){return(
function drag(projection) {
  let v0, q0, r0;

  function dragstarted() {
    v0 = versor.cartesian(projection.invert([d3.event.x, d3.event.y]));
    q0 = versor(r0 = projection.rotate());
  }

  function dragged() {
    const v1 = versor.cartesian(projection.rotate(r0).invert([d3.event.x, d3.event.y]));
    const q1 = versor.multiply(q0, versor.delta(v0, v1));
    projection.rotate(versor.rotation(q1));
  }

  return d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged);
}
)}

function _versor(require){return(
require("versor@0.0.3")
)}

function _34(html){return(
html`
<style>
.draggable {
  cursor: move;
  cursor: -webkit-grab;
  cursor: -moz-grab;
  cursor: grab;
}

.draggable:active {
  cursor: -webkit-grabbing;
  cursor:    -moz-grabbing;
  cursor:         grabbing;
}
</style>`
)}

export default function define(runtime, observer) {
  const main = runtime.module();
  main.variable(observer("Title")).define("Title", ["md"], _Title);
  main.variable(observer()).define(["DOM","width","height","d3","projection","land50","sphere","graticule2","graticule","drag","land110"], _2);
  main.variable(observer()).define(["md"], _3);
  main.variable(observer()).define(["md"], _4);
  main.variable(observer()).define(["width","butterfly","octoformal_octant","sphere","d3","DOM","land50","drag","land110"], _5);
  main.variable(observer()).define(["md"], _6);
  main.variable(observer()).define(["width","butterfly","octoformal_octant","sphere","DOM","d3","land50","graticule2","graticule","drag","land110"], _7);
  main.variable(observer()).define(["md"], _8);
  main.variable(observer()).define(["width","butterfly","octoformal_octant","sphere","DOM","d3","land50","graticule2","graticule","drag","land110"], _9);
  main.variable(observer()).define(["md"], _10);
  main.variable(observer("projection")).define("projection", ["butterfly","octoformal_octant","width","height","sphere"], _projection);
  main.variable(observer("octahedron")).define("octahedron", _octahedron);
  main.variable(observer("octahedronExplicit")).define("octahedronExplicit", _octahedronExplicit);
  main.variable(observer("butterfly")).define("butterfly", ["octahedron","d3"], _butterfly);
  main.variable(observer("octoformal_octant")).define("octoformal_octant", ["d3","cartesian_octoformal"], _octoformal_octant);
  main.variable(observer("cartesian_octoformal")).define("cartesian_octoformal", ["omap","oimap"], _cartesian_octoformal);
  main.variable(observer("omap")).define("omap", ["reval"], _omap);
  main.variable(observer("oimap")).define("oimap", ["reval"], _oimap);
  main.variable(observer("reval")).define("reval", _reval);
  main.variable(observer()).define(["md"], _20);
  main.variable(observer()).define(["md"], _21);
  main.variable(observer("height")).define("height", ["width"], _height);
  main.variable(observer("sphere")).define("sphere", _sphere);
  main.variable(observer("graticule")).define("graticule", ["d3"], _graticule);
  main.variable(observer("graticule2")).define("graticule2", ["d3"], _graticule2);
  main.variable(observer("land50")).define("land50", ["topojson","world50"], _land50);
  main.variable(observer("land110")).define("land110", ["topojson","world110"], _land110);
  main.variable(observer("world50")).define("world50", _world50);
  main.variable(observer("world110")).define("world110", _world110);
  main.variable(observer("topojson")).define("topojson", ["require"], _topojson);
  main.variable(observer("d3")).define("d3", ["require"], _d3);
  main.variable(observer("drag")).define("drag", ["versor","d3"], _drag);
  main.variable(observer("versor")).define("versor", ["require"], _versor);
  main.variable(observer()).define(["html"], _34);
  return main;
}
