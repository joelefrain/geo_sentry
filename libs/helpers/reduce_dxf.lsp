(defun c:EliminaCurvasNo10 ( / ss i ent elev)
  (prompt "\nEliminando curvas de nivel con elevación no múltiplo de 10...")
  (setq ss (ssget '((0 . "LWPOLYLINE")))) ; selecciona todas las polilíneas ligeras
  (if ss
    (progn
      (setq i 0)
      (while (< i (sslength ss))
        (setq ent (ssname ss i))
        (setq elev (cdr (assoc 38 (entget ent)))) ; obtiene la elevación (grupo 38)
        (if (or (not elev) (/= (rem elev 10.0) 0.0))
          (entdel ent) ; elimina si no es múltiplo de 10
        )
        (setq i (1+ i))
      )
      (prompt "\nCurvas no múltiplos de 10 eliminadas.")
    )
    (prompt "\nNo se encontraron polilíneas en el dibujo.")
  )
  (princ)
)
