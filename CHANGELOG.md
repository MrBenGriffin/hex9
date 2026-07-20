### Changelog
```
0.2.0a0 New UUID address mechanism (BREAKING): addresses are now a 128-bit
        UUID with a single-nibble key tail; 0.1.x addresses do not decode.
        Compact uint64 addresses now reach layer 14 (was 13).
        Authalic warp retrained (WGS84 + Sphere) - area deviation now under
        0.1% globally, with the residual at the six octahedral vertices.
        Optional libhex9 C++ backend (~70x faster lat/lon -> address).
        New HexMesh shared-vertex hex grids and a Composition layer-overlay
        class. Distribution trimmed of stale warp data. 464 tests passing.
0.1.3a0 This uses a new UUID (breaking change) mechanism (unreleased).
0.1.2a0 Better PostGIS SQL functions - offering support in PostGIS and QGIS.
        Current focus is now on PROJ9.8 - CRS and projections.
        A few deep bug-fixes. All examples should be working.
        However, many examples are not very interesting!
        A Composition class has been added for overlaying layers.
0.1.1a2 Fixed missing warp file in distro.
0.1.0a6 Warp in place, offering good authalicity.
0.1.0a5 More bug-fixes - especially regarding grids.
        A new example, showing multiple hexagons over Chiginagak volcano, AK.
        A little more documentation.
0.1.0a4 Couple of small bug-fixes,
        A couple of new features
0.1.0a3 Rewrite of the hex address and hex polygon code.
        Hex addresses are not backward compatible, so several files
        (documentation) will need to be revised.
        Examples have all been refactored, and tested accordingly.
0.1.0a2 Improved some documentation; added more licence declr.
0.1.0a1 Modifying Numpy minimum - bitwise_count dependency.
        Removed redundant methods in neighbours.
        hex_layer is still unreliable
0.1.0a0 Initial Package
```