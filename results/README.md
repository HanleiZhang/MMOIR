# Results
Here we provide the results of the models in Mmoir. For supervised models, we have recorded three dimensions of metrics: ID classification, ID+OOD classification, and OOD detection.

<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th rowspan="2">Model Name</th>
      <th rowspan="2">Dialogue Type</th>
      <th colspan="6">ID Classification</th>
      <th colspan="4">ID+OOD Classification</th>
      <th colspan="5">OOD Detection</th>
    </tr>
    <tr>
      <th>Acc</th>
      <th>WF1</th>
      <th>WP</th>
      <th>F1</th>
      <th>P</th>
      <th>R</th>
      <th>ACC</th>
      <th>F1-Known</th>
      <th>F1-Open</th>
      <th>F1</th>
      <th>AUROC</th>
      <th>AUPR-IN</th>
      <th>AUPR-OUT</th>
      <th>FPR95</th>
      <th>DER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MintRec</td>
      <td>MAG_BERT</td>
      <td>Multi</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>MintRec</td>
      <td>MULT</td>
      <td>Multi</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>MintRec</td>
      <td>MAG_BERT</td>
      <td>Single</td>
      <td>72.404</td>
      <td> 72.06 </td>
      <td> 72.938 </td>
      <td>68.286</td>
      <td>68.868</td>
      <td>69.216</td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> 76.628 </td>
        <td> 75.016 </td>
        <td> 75.545 </td>
        <td> 73.78 </td>
        <td> 39.122 </td>
    </tr>
    <tr>
      <td>MintRec</td>
      <td>MULT</td>
      <td>Single</td>
      <td>72.222</td>
      <td> - </td>
      <td> - </td>
      <td>69.394</td>
      <td>70.112</td>
      <td>69.368</td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>MintRec</td>
      <td>MMIM</td>
      <td>Single</td>
      <td>72</td>
      <td>71.964</td>
      <td>72.906</td>
      <td>68.856</td>
      <td>69.72</td>
      <td>69.046</td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td>76.142</td>
        <td>76.392</td>
        <td>773.768</td>
        <td>76.622</td>
        <td>40.538</td>
    </tr>
    <tr>
      <td>MintRec</td>
      <td>SDIF</td>
      <td>Single</td>
      <td>71.642</td>
      <td>71.336</td>
      <td>71.736</td>
      <td>68.188</td>
      <td>69.076</td>
      <td>68.3</td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td>78.608</td>
        <td>77.28</td>
        <td>78.122</td>
        <td>70.578</td>
        <td>37.556</td>
    </tr>
    <tr>
      <td>MintRec</td>
      <td>TCL_MAP</td>
      <td>Single</td>
      <td>73.166</td>
      <td>72.656</td>
      <td> 72.974 </td>
      <td> 68.922 </td>
      <td> 68.9 </td>
      <td> 69.988 </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> 75.208 </td>
        <td> 74.58 </td>
        <td> 73.274 </td>
        <td> 78.888 </td>
        <td> 41.64 </td>
    </tr>
    <tr>
      <td>MintRec2.0</td>
      <td>MAG_BERT</td>
      <td>Multi</td>
      <td> 60.66 </td>
      <td> 59.776 </td>
      <td> 59.89 </td>
      <td> 53.78 </td>
      <td> 55.698 </td>
      <td> 53.938 </td>
      <td> 56.056 </td>
        <td> 47.108 </td>
        <td> 62.334 </td>
        <td> 47.598 </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>MintRec2.0</td>
      <td>MULT</td>
      <td>Multi</td>
      <td> 59.48 </td>
      <td> 59.332 </td>
      <td> 60.036 </td>
      <td> 53.9 </td>
      <td> 54.908 </td>
      <td> 54.148 </td>
      <td> 56.074 </td>
        <td> 46.448 </td>
        <td> 62.928 </td>
        <td> 46.98 </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>MintRec2.0</td>
      <td>MAG_BERT</td>
      <td>Single</td>
      <td> 60.384 </td>
      <td> 59.61 </td>
      <td> 59.996 </td>
      <td> 54.738 </td>
      <td> 57.454 </td>
      <td> 54.538 </td>
      <td> 45.738 </td>
        <td> 46.266 </td>
        <td>39.27 </td>
        <td> 46.04 </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>MintRec2.0</td>
      <td>MULT</td>
      <td>Single</td>
      <td> 60.66 </td>
      <td> 59.546 </td>
      <td> 60.122 </td>
      <td> 54.118 </td>
      <td> 58.016 </td>
      <td> 53.768 </td>
      <td> 46.144 </td>
        <td> 45.646 </td>
        <td> 38.572 </td>
        <td> 45.416 </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>MintRec2.0</td>
      <td>MMIM</td>
      <td>Single</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>MintRec2.0</td>
      <td>SDIF</td>
      <td>Single</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>MintRec2.0</td>
      <td>TCL_MAP</td>
      <td>Single</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>MELD-DA</td>
      <td>MAG_BERT</td>
      <td>Multi</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>MELD-DA</td>
      <td>MULT</td>
      <td>Multi</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>MELD-DA</td>
      <td>MAG_BERT</td>
      <td>Single</td>
      <td> 64.808 </td>
      <td> 63.286 </td>
      <td> 62.994 </td>
      <td> 54.458 </td>
      <td> 57.738 </td>
      <td> 53.704 </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> 68.654 </td>
        <td> 84.86 </td>
        <td> 45.516 </td>
        <td> 77.072 </td>
        <td> 59.256 </td>
    </tr>
    <tr>
      <td>MELD-DA</td>
      <td>MULT</td>
      <td>Single</td>
      <td> 63.35 </td>
      <td> 62.706 </td>
      <td> 62.928 </td>
      <td> 54.79 </td>
      <td> 59.856 </td>
      <td> 54 </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> 66.622 </td>
        <td> 84.33 </td>
        <td> 37.6 </td>
        <td> 85.106 </td>
        <td> 65.298 </td>
    </tr>
    <tr>
      <td>MELD-DA</td>
      <td>MMIM</td>
      <td>Single</td>
      <td> 62.208 </td>
      <td> 60.836</td>
      <td> 61.378 </td>
      <td> 53.166 </td>
      <td> 54.208 </td>
      <td> 54.522 </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> 71.704 </td>
        <td> 86.91 </td>
        <td> 47.63 </td>
        <td> 77.918 </td>
        <td> 59.858 </td>
    </tr>
    <tr>
      <td>MELD-DA</td>
      <td>SDIF</td>
      <td>Single</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>MELD-DA</td>
      <td>TCL_MAP</td>
      <td>Single</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>IEMOCAP-DA</td>
      <td>MAG_BERT</td>
      <td>Multi</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>IEMOCAP-DA</td>
      <td>MULT</td>
      <td>Multi</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>IEMOCAP-DA</td>
      <td>MAG_BERT</td>
      <td>Single</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>IEMOCAP-DA</td>
      <td>MULT</td>
      <td>Single</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>IEMOCAP-DA</td>
      <td>MMIM</td>
      <td>Single</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>IEMOCAP-DA</td>
      <td>SDIF</td>
      <td>Single</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
    <tr>
      <td>IEMOCAP-DA</td>
      <td>TCL_MAP</td>
      <td>Single</td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
      <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
    </tr>
  </tbody>
</table>


For unsupervised models:
| Dataset    | Model Name    | Dialogue Type | NMI  | ARI | ACC | fmi |
|------------|---------------|---------------|------|-------------|----------|---------|
| MintRec    | MCN           | Single        | 19.892    | 2.33           | 17.978        | 9.408       |
| MintRec    | CC            | Single        | 47.142    | 22.206           | 42.024        | 27.006       |
| MintRec    | sccl          | Single        | 45.04    | 14.718           | 37.394        | 24.098       |
| MintRec    | USNID         | Single        | -    | -           | -        | -       |
| MintRec    | UMC           | Single        | -    | -           | -        | -       |
| MintRec2.0 | MCN           | Single        | -    | -           | -        | -       |
| MintRec2.0 | CC            | Single        | -    | -           | -        | -       |
| MintRec2.0 | sccl          | Single        | -    | -           | -        | -       |
| MintRec2.0 | USNID         | Single        | -    | -           | -        | -       |
| MintRec2.0 | UMC           | Single        | -    | -           | -        | -       |
| MELD-DA    | MCN           | Single        | -    | -           | -        | -       |
| MELD-DA    | CC            | Single        | -    | -           | -        | -       |
| MELD-DA    | sccl          | Single        | -    | -           | -        | -       |
| MELD-DA    | USNID         | Single        | -    | -           | -        | -       |
| MELD-DA    | UMC           | Single        | -    | -           | -        | -       |
| IEMOCAP-DA | MCN           | Single        | -    | -           | -        | -       |
| IEMOCAP-DA | CC            | Single        | -    | -           | -        | -       |
| IEMOCAP-DA | sccl          | Single        | -    | -           | -        | -       |
| IEMOCAP-DA | USNID         | Single        | -    | -           | -        | -       |
| IEMOCAP-DA | UMC           | Single        | -    | -           | -        | -       |


