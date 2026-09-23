if (typeof dwr == 'undefined' || dwr.engine == undefined) throw new Error('You must include DWR engine before including this file');

(function() {
  if (dwr.engine._getObject("lgdDwrDistrictService") == undefined) {
    var p;
    
    p = {};

    /**
     * @param {int} p0 a param
     * @param {class java.lang.Integer} p1 a param
     * @param {class java.lang.Integer} p2 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictListbyStateCodeForLocalBody = function(p0, p1, p2, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictListbyStateCodeForLocalBody', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.bean.GovernmentOrder} p0 a param
     * @param {class in.nic.pes.lgd.forms.DistrictForm} p1 a param
     * @param {interface java.util.List} p2 a param
     * @param {interface org.hibernate.Session} p3 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveDataInAttachment = function(p0, p1, p2, p3, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveDataInAttachment', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.DistrictForm} p0 a param
     * @param {interface java.util.List} p1 a param
     * @param {interface org.hibernate.Session} p3 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveDataInMap = function(p0, p1, p3, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveDataInMap', arguments);
    };

    /**
     * @param {class java.lang.String} p0 a param
     * @param {class java.lang.String} p1 a param
     * @param {class [Ljava.lang.String;} p2 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getVillageList = function(p0, p1, p2, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getVillageList', arguments);
    };

    /**
     * @param {class java.lang.String} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getVillageList = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getVillageList', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.DistrictForm} p0 a param
     * @param {interface org.hibernate.Session} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveDataInGovtOrder = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveDataInGovtOrder', arguments);
    };

    /**
     * @param {interface java.util.List} p0 a param
     * @param {class java.lang.Integer} p1 a param
     * @param {interface org.hibernate.Session} p2 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveDataInAttachDraftDistrict = function(p0, p1, p2, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveDataInAttachDraftDistrict', arguments);
    };

    /**
     * @param {class java.lang.String} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getSubdistrictListbyDistrict = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getSubdistrictListbyDistrict', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {class java.lang.String} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictListbyStateCode = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictListbyStateCode', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictListbyStateCode = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictListbyStateCode', arguments);
    };

    /**
     * @param {class java.lang.String} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictViewList = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictViewList', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.DistrictForm} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveDistrict = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveDistrict', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {int} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getTargetDistrictList = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getTargetDistrictList', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictDetailsModify = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictDetailsModify', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.DistrictForm} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.modifyDistrictInfo = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'modifyDistrictInfo', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.DistrictForm} p0 a param
     * @param {int} p1 a param
     * @param {int} p2 a param
     * @param {int} p3 a param
     * @param {interface org.hibernate.Session} p4 a param
     * @param {function|Object} callback callback function or options object
     */
    p.publishDistrict = function(p0, p1, p2, p3, p4, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'publishDistrict', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.StandardCodeForm} p0 a param
     * @param {class java.lang.String} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.updateStandardCode = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'updateStandardCode', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.StandardCodeForm} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getStandardCode = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getStandardCode', arguments);
    };

    /**
     * @param {char} p0 a param
     * @param {int} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictHistoryDetail = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictHistoryDetail', arguments);
    };

    /**
     * @param {char} p0 a param
     * @param {int} p1 a param
     * @param {int} p2 a param
     * @param {int} p3 a param
     * @param {function|Object} callback callback function or options object
     */
    p.GovOrderDetail = function(p0, p1, p2, p3, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'GovOrderDetail', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictListbyStateCodeForListBox = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictListbyStateCodeForListBox', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.DistrictForm} p2 a param
     * @param {interface org.hibernate.Session} p3 a param
     * @param {function|Object} callback callback function or options object
     */
    p.insertDistrict = function(p2, p3, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'insertDistrict', arguments);
    };

    /**
     * @param {class java.lang.String} p0 a param
     * @param {class java.lang.Integer} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.findDuplicate = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'findDuplicate', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {int} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getTargetDistrictShiftSDistrict = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getTargetDistrictShiftSDistrict', arguments);
    };

    /**
     * @param {class java.lang.Integer} p0 a param
     * @param {class java.lang.Integer} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictListExtend = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictListExtend', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getStateVersion = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getStateVersion', arguments);
    };

    /**
     * @param {class java.lang.Integer} p0 a param
     * @param {class java.lang.Integer} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictListExtendforBlock = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictListExtendforBlock', arguments);
    };

    /**
     * @param {class java.lang.Integer} p0 a param
     * @param {class java.lang.String} p1 a param
     * @param {class java.lang.Integer} p2 a param
     * @param {char} p3 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictListbyStateCodeExtend = function(p0, p1, p2, p3, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictListbyStateCodeExtend', arguments);
    };

    /**
     * @param {interface org.hibernate.Session} p0 a param
     * @param {class [I} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.updateDistrictMap = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'updateDistrictMap', arguments);
    };

    /**
     * @param {class java.lang.Integer} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.DistrictWiseLBReportDetail = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'DistrictWiseLBReportDetail', arguments);
    };

    /**
     * @param {class java.lang.Integer} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictNameEnglishbyDistrictCode = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictNameEnglishbyDistrictCode', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {int} p1 a param
     * @param {int} p2 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictListbyStateCodeForListBoxExtended = function(p0, p1, p2, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictListbyStateCodeForListBoxExtended', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {class java.lang.String} p1 a param
     * @param {int} p2 a param
     * @param {int} p3 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictListbyStateCodeExtended = function(p0, p1, p2, p3, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictListbyStateCodeExtended', arguments);
    };

    /**
     * @param {class java.lang.String} p0 a param
     * @param {class java.lang.Integer} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.updatePesaStatus = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'updatePesaStatus', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {class java.lang.Integer} p1 a param
     * @param {class java.lang.Integer} p2 a param
     * @param {class java.lang.Integer} p3 a param
     * @param {class java.lang.Character} p4 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictListbyStateCodeForLocalBodyExceptEntity = function(p0, p1, p2, p3, p4, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictListbyStateCodeForLocalBodyExceptEntity', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {int} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getTargetDistrictShiftSDistrictForlocalbody = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getTargetDistrictShiftSDistrictForlocalbody', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {class java.lang.String} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictListbyStateCodeForLocalbody = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictListbyStateCodeForLocalbody', arguments);
    };

    /**
     * @param {char} p0 a param
     * @param {int} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictHistoryReport = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictHistoryReport', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictListbyDistrictCodeForLocalBody = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictListbyDistrictCodeForLocalBody', arguments);
    };

    /**
     * @param {class java.lang.String} p0 a param
     * @param {class java.lang.Character} p1 a param
     * @param {class java.lang.Character} p2 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getHeirarchyByParentCodes = function(p0, p1, p2, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getHeirarchyByParentCodes', arguments);
    };

    /**
     * @param {class java.lang.String} p0 a param
     * @param {class java.lang.Integer} p1 a param
     * @param {class java.lang.Integer} p2 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictListbyStates = function(p0, p1, p2, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictListbyStates', arguments);
    };

    /**
     * @param {class java.lang.Integer} p0 a param
     * @param {class java.lang.String} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictListbyCriteria = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictListbyCriteria', arguments);
    };

    /**
     * @param {class java.lang.Integer} p0 a param
     * @param {class java.lang.Integer} p1 a param
     * @param {class java.util.Date} p2 a param
     * @param {function|Object} callback callback function or options object
     */
    p.distInvalFnAfterCreateMulDist = function(p0, p1, p2, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'distInvalFnAfterCreateMulDist', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.DistrictForm} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveEffectiveDateEntityDistrict = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveEffectiveDateEntityDistrict', arguments);
    };

    /**
     * @param {class java.lang.Integer} p0 a param
     * @param {class java.lang.Character} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getISPESAENTITY = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getISPESAENTITY', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.PesaEntityForm} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.savePesaMapping = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'savePesaMapping', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.StandardCodeForm} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getStandardCodePostal = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getStandardCodePostal', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.draft.form.RevenueDraftForm} p0 a param
     * @param {interface java.util.List} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveDistrictinDraft = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveDistrictinDraft', arguments);
    };

    /**
     * @param {class java.lang.String} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictListbydistricts = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictListbydistricts', arguments);
    };

    /**
     * @param {char} p0 a param
     * @param {int} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getEntityHistoryDetail = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getEntityHistoryDetail', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.DistrictForm} p0 a param
     * @param {int} p1 a param
     * @param {int} p2 a param
     * @param {interface org.hibernate.Session} p3 a param
     * @param {function|Object} callback callback function or options object
     */
    p.publishMapLandRegion = function(p0, p1, p2, p3, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'publishMapLandRegion', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.DistrictForm} p0 a param
     * @param {int} p1 a param
     * @param {int} p2 a param
     * @param {int} p3 a param
     * @param {interface org.hibernate.Session} p4 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveHeadquarters = function(p0, p1, p2, p3, p4, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveHeadquarters', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.SubDistrictForm} p0 a param
     * @param {int} p1 a param
     * @param {int} p2 a param
     * @param {int} p3 a param
     * @param {interface org.hibernate.Session} p4 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveHeadquarters = function(p0, p1, p2, p3, p4, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveHeadquarters', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {int} p1 a param
     * @param {int} p2 a param
     * @param {int} p3 a param
     * @param {interface org.hibernate.Session} p4 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveReplacedBy = function(p0, p1, p2, p3, p4, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveReplacedBy', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {int} p1 a param
     * @param {int} p2 a param
     * @param {int} p3 a param
     * @param {interface org.hibernate.Session} p4 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveReplaces = function(p0, p1, p2, p3, p4, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveReplaces', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {int} p1 a param
     * @param {int} p2 a param
     * @param {interface org.hibernate.Session} p3 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveNewSubdistritVersion = function(p0, p1, p2, p3, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveNewSubdistritVersion', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {int} p1 a param
     * @param {int} p2 a param
     * @param {int} p3 a param
     * @param {interface org.hibernate.Session} p4 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveNewDistrictVersion = function(p0, p1, p2, p3, p4, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveNewDistrictVersion', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {int} p1 a param
     * @param {int} p2 a param
     * @param {int} p3 a param
     * @param {interface org.hibernate.Session} p4 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveNewSubdistrictVersionPartContri = function(p0, p1, p2, p3, p4, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveNewSubdistrictVersionPartContri', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {int} p1 a param
     * @param {int} p2 a param
     * @param {interface org.hibernate.Session} p3 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveNewSubdistritfordist = function(p0, p1, p2, p3, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveNewSubdistritfordist', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {int} p1 a param
     * @param {int} p2 a param
     * @param {interface org.hibernate.Session} p3 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveNewSubdist = function(p0, p1, p2, p3, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveNewSubdist', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {int} p1 a param
     * @param {int} p2 a param
     * @param {int} p3 a param
     * @param {int} p4 a param
     * @param {int} p5 a param
     * @param {interface org.hibernate.Session} p6 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveNewvillage = function(p0, p1, p2, p3, p4, p5, p6, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveNewvillage', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {int} p1 a param
     * @param {interface org.hibernate.Session} p2 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveNewVillageVersion = function(p0, p1, p2, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveNewVillageVersion', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getSubdistrictDetailsModify = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getSubdistrictDetailsModify', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictListByDistCode = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictListByDistCode', arguments);
    };

    /**
     * @param {class java.lang.String} p0 a param
     * @param {class java.lang.Integer} p1 a param
     * @param {class java.lang.Integer} p2 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getSubDistrictListbyDistrictForLocalBody = function(p0, p1, p2, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getSubDistrictListbyDistrictForLocalBody', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.DistrictForm} p0 a param
     * @param {interface org.hibernate.Session} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.publishSubdistrit = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'publishSubdistrit', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {int} p1 a param
     * @param {int} p2 a param
     * @param {interface org.hibernate.Session} p3 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveNewSubDistrictVersion = function(p0, p1, p2, p3, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveNewSubDistrictVersion', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {int} p1 a param
     * @param {int} p2 a param
     * @param {interface org.hibernate.Session} p3 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveNewSubdistrit = function(p0, p1, p2, p3, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveNewSubdistrit', arguments);
    };

    /**
     * @param {interface java.util.List} p0 a param
     * @param {interface org.hibernate.Session} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveNewvillver = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveNewvillver', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {int} p1 a param
     * @param {interface org.hibernate.Session} p2 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveNewvill = function(p0, p1, p2, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveNewvill', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getSubDistrictVersion = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getSubDistrictVersion', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictListGlobal = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictListGlobal', arguments);
    };

    /**
     * @param {class java.lang.String} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getMergeDistrictWithSubDistrictWithSubdistrict = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getMergeDistrictWithSubDistrictWithSubdistrict', arguments);
    };

    /**
     * @param {class java.lang.String} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getPartSubdistrictListbyDistrict = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getPartSubdistrictListbyDistrict', arguments);
    };

    /**
     * @param {function|Object} callback callback function or options object
     */
    p.getGazettePublicationDate = function(callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getGazettePublicationDate', arguments);
    };

    /**
     * @param {class java.lang.String} p0 a param
     * @param {class java.util.Date} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getGazettePublicationDateSave = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getGazettePublicationDateSave', arguments);
    };

    /**
     * @param {class java.lang.Integer} p0 a param
     * @param {char} p1 a param
     * @param {class java.lang.Integer} p2 a param
     * @param {class java.lang.Integer} p3 a param
     * @param {class java.lang.Integer} p4 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getPcCode = function(p0, p1, p2, p3, p4, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getPcCode', arguments);
    };

    /**
     * @param {interface org.springframework.web.multipart.MultipartFile} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.writeMap = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'writeMap', arguments);
    };

    /**
     * @param {class java.lang.String} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getSubDistrictViewList = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getSubDistrictViewList', arguments);
    };

    /**
     * @param {class java.lang.String} p0 a param
     * @param {class java.lang.String} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.invalidateLoop = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'invalidateLoop', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.DistrictForm} p0 a param
     * @param {class in.nic.pes.lgd.forms.GovernmentOrderForm} p1 a param
     * @param {interface java.util.List} p2 a param
     * @param {function|Object} callback callback function or options object
     */
    p.changeDistrict = function(p0, p1, p2, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'changeDistrict', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.DistrictForm} p0 a param
     * @param {class in.nic.pes.lgd.forms.GovernmentOrderForm} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.changeDistrictforTemplate = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'changeDistrictforTemplate', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.DistrictForm} p0 a param
     * @param {interface java.util.List} p2 a param
     * @param {interface java.util.List} p3 a param
     * @param {boolean} p4 a param
     * @param {class [Ljava.lang.String;} p5 a param
     * @param {function|Object} callback callback function or options object
     */
    p.modifyDistrictCrInfo = function(p0, p2, p3, p4, p5, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'modifyDistrictCrInfo', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getAttachmentsbyOrderCode = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getAttachmentsbyOrderCode', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.DistrictForm} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveInvalidateDistrict = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveInvalidateDistrict', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getSubdistrictListbyDistrictCode = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getSubdistrictListbyDistrictCode', arguments);
    };

    /**
     * @param {class java.lang.String} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getSubdistrictListbyDistrictInDistrictForLocalBody = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getSubdistrictListbyDistrictInDistrictForLocalBody', arguments);
    };

    /**
     * @param {class java.lang.String} p0 a param
     * @param {class java.lang.String} p1 a param
     * @param {class [Ljava.lang.String;} p2 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getVillageListRemovingOldVillageForDistrictForm = function(p0, p1, p2, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getVillageListRemovingOldVillageForDistrictForm', arguments);
    };

    /**
     * @param {class java.lang.Integer} p0 a param
     * @param {class java.lang.String} p1 a param
     * @param {class java.lang.String} p2 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictForGIS = function(p0, p1, p2, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictForGIS', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.draft.form.RevenueDraftForm} p0 a param
     * @param {interface java.util.List} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveDistrictRenameDraft = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveDistrictRenameDraft', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.draft.form.RevenueDraftForm} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveDistrictRenamePublish = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveDistrictRenamePublish', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.draft.form.RevenueDraftForm} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.saveDistrictInvalidatePublish = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'saveDistrictInvalidatePublish', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {int} p1 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictListbyStateCodeExceptDistrict = function(p0, p1, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictListbyStateCodeExceptDistrict', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrictList = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrictList', arguments);
    };

    /**
     * @param {int} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrict = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrict', arguments);
    };

    /**
     * @param {class java.lang.String} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.getDistrict = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'getDistrict', arguments);
    };

    /**
     * @param {class in.nic.pes.lgd.forms.SubDistrictForm} p0 a param
     * @param {function|Object} callback callback function or options object
     */
    p.save = function(p0, callback) {
      return dwr.engine._execute(p._path, 'lgdDwrDistrictService', 'save', arguments);
    };
    
    dwr.engine._setObject("lgdDwrDistrictService", p);
  }
})();

