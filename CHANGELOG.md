### Changelog
```
0.3.0a1   One projection chain (BREAKING): the classic per-ellipsoid regime is
          retired and set_ellipsoid() no longer takes via_sphere. All addressing
          runs through the unit authalic sphere, so one trained field serves
          every ellipsoid. The two chains minted different addresses below
          layer 5 with nothing in the address to say which; only the via-sphere
          ones remain valid.
          libhex9 acceleration works again, and now needs libhex9 2.0.0+: 2.0.0
          dropped the regime toggle hhg9 had been probing for, so the backend
          had been silently falling back to pure Python (~200x slower on
          lat/lon -> b_oct). Parity with the Python reference is 1.7e-15.
          Fixed: a region spanning a meridian seam rendered empty under the
          dynamic local net - octant corner coincidence was tested in
          (lat, lon), where a pole corner has arbitrary longitude.
          Obsolete reference-engine tests removed. 532 tests passing.

0.3.0a0 New authalising warp (BREAKING):
        sphere symmetric warp + authalic latitude substantially reduces
        remaining area deficit. Now no area is >±1% equal-area, and only
        252 cells at L5 are >±0.1% (all close to the six octahedral vertices).
        hex9 tends towards ownership over lineage and this is now more explicit.
        Moore/Hilbert curve-class address is now implemented at O(1) from h9 uuid.
        curve addresses are prefixed 'c'; eg 'c1157461-0355-4541-5530-153322584823'
        inversion (conversion from curve to hex9) is slower - and not recommended
        except for round-trip testing. 538 tests passing [100%].
        rendering, imagery, composition, and other functions have been improved.
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