#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import time
import sys
import xmltodict
import ismrmrd
import lxml.etree as ET
# import logging

sys.path.append('/home/zzgroup/Desktop/umr_rawdata/UIHrawdata2h5/ismrmrd-python-master')
import os

import scipy.io


def Read_UIH_DHL(fid):
    DHL = {}
    DHL['m_ucVersion'] = np.fromfile(fid, dtype=np.uint8, count=1)
    fid.seek(1, 1)
    DHL['m_uReceiveGain'] = np.fromfile(fid, dtype=np.uint8, count=1)
    fid.seek(1, 1)
    DHL['m_ulDataLength'] = np.fromfile(fid, dtype=np.uint32, count=1)
    DHL['m_ullMeasUID'] = np.fromfile(fid, dtype=np.uint64, count=1)
    DHL['m_ulSysTimeStamp'] = np.fromfile(fid, dtype=np.uint32, count=1)
    DHL['m_ulScanTimeStamp'] = np.fromfile(fid, dtype=np.uint32, count=1)
    DHL['m_ulVSMTimeStamp'] = np.fromfile(fid, dtype=np.uint32, count=1)
    DHL['m_lTabPositionX'] = np.fromfile(fid, dtype=np.int32, count=1)
    DHL['m_lTabPositionY'] = np.fromfile(fid, dtype=np.int32, count=1)
    DHL['m_lTabPositionZ'] = np.fromfile(fid, dtype=np.int32, count=1)
    DHL['m_ushSamples'] = np.fromfile(fid, dtype=np.uint16, count=1)
    DHL['m_ushUsedChannels'] = np.fromfile(fid, dtype=np.uint16, count=1)
    fid.seek(4, 1)
    DHL['m_ullCtrlFlags'] = np.fromfile(fid, dtype=np.uint64, count=1)
    DHL['m_ushRepeat'] = np.fromfile(fid, dtype=np.uint16, count=1)
    DHL['m_ushCardiacPhase'] = np.fromfile(fid, dtype=np.uint16, count=1)
    DHL['m_ushSlice'] = np.fromfile(fid, dtype=np.uint16, count=1)
    DHL['m_ushAverage'] = np.fromfile(fid, dtype=np.uint16, count=1)
    DHL['m_ushContrast'] = np.fromfile(fid, dtype=np.uint16, count=1)
    DHL['m_ushSet'] = np.fromfile(fid, dtype=np.uint16, count=1)
    DHL['m_ushShot'] = np.fromfile(fid, dtype=np.uint16, count=1)
    DHL['m_ushSegment'] = np.fromfile(fid, dtype=np.uint16, count=1)
    DHL['m_ushPELine'] = np.fromfile(fid, dtype=np.uint16, count=1)
    DHL['m_ushSPELine'] = np.fromfile(fid, dtype=np.uint16, count=1)
    DHL['m_ushUserCtrlFlag'] = np.fromfile(fid, dtype=np.uint16, count=1)
    DHL['m_ushUser'] = np.fromfile(fid, dtype=np.uint16, count=5)
    DHL['m_ushCutoffHead'] = np.fromfile(fid, dtype=np.uint16, count=1)
    DHL['m_ushCutoffTail'] = np.fromfile(fid, dtype=np.uint16, count=1)
    DHL['m_ushROCenter'] = np.fromfile(fid, dtype=np.uint16, count=1)
    DHL['m_ushPECenter'] = np.fromfile(fid, dtype=np.uint16, count=1)
    DHL['m_ushSPECenter'] = np.fromfile(fid, dtype=np.uint16, count=1)
    DHL['m_aflPosition'] = np.fromfile(fid, dtype=np.float32, count=3)
    DHL['m_aflOrientation'] = np.fromfile(fid, dtype=np.float32, count=4)

    fid.seek(48, 1)
    fid.seek(14, 1)
    fid.seek(4, 1)

    DHL['m_ushUser'].shape = [1, 5]
    DHL['m_aflPosition'].shape = [1, 3]
    DHL['m_aflOrientation'].shape = [1, 4]
    # DHL['DHLMask']=DHLFlag(DHL['m_ullCtrlFlags'])

    return DHL


def DHLFlag(value):
    DHLMask = {}
    # define DHL_ACQUISITION_END       SET_BIT_AT(0)  // set by RT, the follows set by sequence
    DHLMask['DHL_ACQUISITION_END'] = np.bitwise_and(value, 2 ** 0)
    # define DHL_FIRST_LINE_ACQ        SET_BIT_AT(1)
    DHLMask['DHL_FIRST_LINE_ACQ'] = np.bitwise_and(value, 2 ** 1)
    # define DHL_LAST_LINE_ACQ         SET_BIT_AT(2)
    DHLMask['DHL_LAST_LINE_ACQ'] = np.bitwise_and(value, 2 ** 2)
    # define DHL_FIRST_LINE_SLICE      SET_BIT_AT(3)
    DHLMask['DHL_FIRST_LINE_SLICE'] = np.bitwise_and(value, 2 ** 3)
    # define DHL_LAST_LINE_SLICE       SET_BIT_AT(4)
    DHLMask['DHL_LAST_LINE_SLICE'] = np.bitwise_and(value, 2 ** 4)
    # define DHL_FIRST_LINE_SECTION    SET_BIT_AT(5)
    DHLMask['DHL_FIRST_LINE_SECTION'] = np.bitwise_and(value, 2 ** 5)
    # define DHL_LAST_LINE_SECTION     SET_BIT_AT(6)
    DHLMask['DHL_LAST_LINE_SECTION'] = np.bitwise_and(value, 2 ** 6)
    # define DHL_PHASE_CORRECTION      SET_BIT_AT(7)
    DHLMask['DHL_PHASE_CORRECTION'] = np.bitwise_and(value, 2 ** 7)
    # define DHL_READOUT_REVERSION     SET_BIT_AT(8)  // set kx acquisition direction flag in EPI
    DHLMask['DHL_READOUT_REVERSION'] = np.bitwise_and(value, 2 ** 8)
    # define DHL_PPA_REFLINE           SET_BIT_AT(9)
    DHLMask['DHL_PPA_REFLINE'] = np.bitwise_and(value, 2 ** 9)
    # define DHL_FEEDBACK              SET_BIT_AT(10)
    DHLMask['DHL_FEEDBACK'] = np.bitwise_and(value, 2 ** 10)
    # define DHL_FIRST_LINE_SPE        SET_BIT_AT(11
    DHLMask['DHL_FIRST_LINE_SPE'] = np.bitwise_and(value, 2 ** 11)
    # define DHL_LAST_LINE_SPE         SET_BIT_AT(12)
    DHLMask['DHL_LAST_LINE_SPE'] = np.bitwise_and(value, 2 ** 12)
    # define DHL_NOISE_SCAN            SET_BIT_AT(24)
    DHLMask['DHL_NOISE_SCAN'] = np.bitwise_and(value, 2 ** 24);
    # define DHL_SEQUENCE_STOP         SET_BIT_AT(63)
    # DHLMask.DHL_SEQUENCE_STOP           = bitand(value, 2^63);

    return DHLMask


def Read_UIH_ADC(fid, DHL):
    iSamples = DHL['m_ushSamples'][0]
    iChannels = DHL['m_ushUsedChannels'][0]
    ReceiveGain = 1
    ADCdata = []
    if DHL['m_uReceiveGain'] == 1:
        ReceiveGain = 1.99
    if DHL['m_ucVersion'] == 1:
        ADCdata = np.fromfile(fid, dtype=np.float32, count=iSamples * iChannels * 2)
        ADCdata.shape = 2, iSamples * iChannels
        ADCdata = ADCdata[0, :] + ADCdata[1, :] * 1j
        ADCdata.shape = iSamples, iChannels
    elif DHL['m_ucVersion'] == 2 or DHL['m_ucVersion'] == 3:
        ADCdata = np.fromfile(fid, dtype=np.float32, count=(2 + iSamples) * iChannels * 2)
        ADCdata.shape = 2, (2 + iSamples), iChannels
        ADCdata = ADCdata[0, :] + ADCdata[1, :] * 1j
        ADCdata = ADCdata[2:, :]
        ADCdata.shape = iSamples, iChannels

    return ADCdata


def parse(filename):
    filedir, namestr = os.path.split(filename)
    namestr, ext = os.path.split(namestr)
    xmlfile = os.path.join(filedir, namestr, '.prot')
    xmlfid = open(xmlfile, 'w')
    fid = open(filename, 'r')
    protoffset = np.fromfile(fid, dtype=np.uint32, count=1)
    xmlcontent = fid.read(protoffset[0])
    xmlfid.write(xmlcontent)
    xmlfid.close()
    fid.close()
    prot_dict = xmltodict.parse(xmlcontent)
    print('Extract protocal file: ' + xmlfile + ' from the data')
    fid = open(filename, 'rb')
    fid.seek(4 + protoffset[0], 0)

    ACQFlag = 0
    ADCIndex = 0;
    DHLall = {}
    rawdata = []
    while ACQFlag < 1:
        DHL = Read_UIH_DHL(fid)
        DHLMask = DHLFlag(DHL['m_ullCtrlFlags'])
        ACQFlag = DHLMask['DHL_ACQUISITION_END'][0]
        if ACQFlag == 1:
            break
        if ADCIndex == 0:
            for keys, values in DHL.items():
                DHLall[keys] = values
        else:
            for keys, values in DHL.items():
                DHLall[keys] = np.concatenate((DHLall[keys], values), axis=0)

        ADCdata1 = Read_UIH_ADC(fid, DHL)
        rawdata.append(ADCdata1)
        ADCIndex = ADCIndex + 1;
    fid.close()

    return DHLall, rawdata, prot_dict


def UIH2Acquisition(DHL, data, prot_dict, idx):
    acqblock = ismrmrd.Acquisition.from_array(data[idx].swapaxes(1, 0))
    DHLMask = DHLFlag(np.int(DHL['m_ullCtrlFlags'][idx]))

    acqblock.__set_version(DHL['m_ucVersion'][idx])
    acqblock.__set_measurement_uid(DHL['m_ullMeasUID'][idx])
    acqblock.__set_scan_counter(0)
    acqblock.__set_acquisition_time_stamp(DHL['m_ulSysTimeStamp'][idx])
    #acqblock.__set_physiology_time_stamp(DHL['m_ulVSMTimeStamp'][idx])
    #acqblock.__set_number_of_samples(DHL['m_ushSamples'][idx])
    acqblock.__set_available_channels(DHL['m_ushUsedChannels'][idx])
    #acqblock.__set_active_channels(DHL['m_ushUsedChannels'][idx])
    #acqblock.__set_channel_mask(0)
    acqblock.__set_discard_pre(DHL['m_ushCutoffHead'][idx])
    acqblock.__set_discard_post(DHL['m_ushCutoffTail'][idx])
    acqblock.__set_center_sample(DHL['m_ushROCenter'][idx] * 2)
    acqblock.__set_encoding_space_ref(0)
    #acqblock.__set_trajectory_dimensions(0)

    uih_prot_root = prot_dict['UProtocol']['Root']['Seq']
    NoiseScan_Dwelltime = int(uih_prot_root['NoiseScan']['Dwelltime']['Value'])
    ImagingScan_Dwelltime = int(uih_prot_root['Basic']['Dwelltime']['Value'])
    if DHLMask['DHL_NOISE_SCAN']:
        acqblock.__set_sample_time_us(NoiseScan_Dwelltime / 1000)
    else:
        acqblock.__set_sample_time_us(ImagingScan_Dwelltime / 1000)

    # Slice position information
    DHL_m_aflPosition = DHL['m_aflPosition'][idx, :]
    acqblock.__set_position((DHL_m_aflPosition[0],DHL_m_aflPosition[1],DHL_m_aflPosition[2]))

    DHL_patient_table_position = [DHL['m_lTabPositionX'][idx], DHL['m_lTabPositionY'][idx], DHL['m_lTabPositionZ'][idx]]
    acqblock.__set_patient_table_position((DHL_patient_table_position[0]*0.1,DHL_patient_table_position[1]*0.1,DHL_patient_table_position[2]*0.1))

    # Encoding loop Counters
    acqblock.idx.kspace_encode_step_1   = DHL['m_ushPELine'][idx]
    acqblock.idx.kspace_encode_step_2   = DHL['m_ushSPELine'][idx]
    acqblock.idx.average                = DHL['m_ushAverage'][idx]
    acqblock.idx.slice                  = DHL['m_ushSlice'][idx]
    acqblock.idx.contrast               = DHL['m_ushContrast'][idx]
    acqblock.idx.phase                  = DHL['m_ushCardiacPhase'][idx]
    acqblock.idx.repetition             = DHL['m_ushRepeat'][idx]
    acqblock.idx.set                    = DHL['m_ushSet'][idx]
    acqblock.idx.segment                = DHL['m_ushSegment'][idx]

    # Acquisition Flags
    acqblock.clear_all_flags()

    if DHLMask['DHL_NOISE_SCAN']:
        acqblock.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)

    if DHLMask['DHL_READOUT_REVERSION']:
        acqblock.setFlag(ismrmrd.ACQ_IS_REVERSE)

    if DHLMask['DHL_PHASE_CORRECTION']:
        acqblock.setFlag(ismrmrd.ACQ_IS_PHASECORR_DATA)

    if DHLMask['DHL_LAST_LINE_ACQ']:
       acqblock.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)

    if DHLMask['DHL_FIRST_LINE_SPE']:
        acqblock.setFlag(ismrmrd.ACQ_FIRST_IN_ENCODE_STEP2)

    if DHLMask['DHL_LAST_LINE_SPE']:
        acqblock.setFlag(ismrmrd.ACQ_LAST_IN_ENCODE_STEP2)

    if DHLMask['DHL_FIRST_LINE_SLICE']:
        acqblock.setFlag(ismrmrd.ACQ_FIRST_IN_SLICE)

    if DHLMask['DHL_LAST_LINE_SLICE']:
        acqblock.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)

    return acqblock

def UIH2ISMRMRD(filename):
    filedir, namestr0 = os.path.split(filename)
    namestr, ext = os.path.splitext(namestr0)
    xmlfile = os.path.join(filedir, namestr+ '.prot')
    h5file=os.path.join(filedir, namestr+'_ismrmrd.h5')
    h5file1=os.path.join(filedir, namestr+'_ismrmrd1.h5')
    DHLall, rawdata, prot_dict=parse(filename)
    xslt_root = ET.XML(xslt_prot2ismrmrdxml)
    transform=ET.XSLT(xslt_root)
    protdoc = ET.parse(xmlfile)
    headxml=transform(protdoc)
    header=ismrmrd.xsd.CreateFromDocument(headxml)

    dset=ismrmrd.Dataset(h5file)
    for idx in range(len(rawdata)):
        acqblock=UIH2Acquisition(DHLall, rawdata, prot_dict, idx)
        dset.append_acquisition(acqblock)
    dset.close


   # with ismrmrd.File(h5file1) as file:
  #      dataset=file['dataset']
   #     dataset.header=header
    #    dataset.acquisitions=dset








xslt_prot2ismrmrdxml="""<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
	<xsl:param name="convert_parameter" select="default"/>
	<!-- TODO read from patient parameter and pass as xslt param ? -->
	<xsl:param name="patient_position">HFP</xsl:param>
	<xsl:output method="xml" indent="yes"/>
	<xsl:template match="/">
		<!-- const begin -->
    
    <!-- Encoding Mode 2D, Multislice 2D or 3D -->
		<xsl:variable name="kDIMENSION_2D">1</xsl:variable>
		<xsl:variable name="kDIMENSION_3D">2</xsl:variable>
    <xsl:variable name="notMultiSliceMode">0</xsl:variable>
   
    <!-- Recon 2D (RO-PE plane) kspace interpolation 1x, 2x, 1.5x  -->
    <xsl:variable name="kINT_2x">1</xsl:variable>
    <xsl:variable name="kINT_1x">0</xsl:variable>

    <!-- Fast imaging methods: no, Fast1D, Fast2D, Fast2DT, uCS2D, 4DMRA -->
	  <xsl:variable name="kPPAMethod_None">0</xsl:variable>
    <xsl:variable name="kPPAMethod_Fast1D">1</xsl:variable>
	  <xsl:variable name="kPPAMethod_Fast2D">10</xsl:variable>
	  <xsl:variable name="kPPAMethod_uCS">11</xsl:variable>
    <xsl:variable name="kPPAMethod_Fast2DT">20</xsl:variable>
    <xsl:variable name="kPPAMethod_4DMRA">33</xsl:variable>
    
		<!-- const end -->
		
		<!-- variable begin -->
  
    <!-- EncodingMode 2D or 3D -->
    <xsl:variable name="is_3D"                   select="/UProtocol/Root/IRIP/FromSeq/KSpace/EncodingMode3D/Value" />
    <xsl:variable name="dimension" 			         select="/UProtocol/Root/Seq/KSpace/Dimension/Value" />
    
    <!-- EncodingSpace 1st dimension - Readout(RO) -->
		<xsl:variable name="matrix_ro_ui" 		       select="/UProtocol/Root/Seq/KSpace/MatrixRO/Value" />
    <xsl:variable name="matrix_ro_real" 	       select="matrix_ro_ui*2" />
    
    <!-- EncodingSpace 2nd dimension - PhaseEncoding(PE) -->
		<xsl:variable name="matrix_pe_ui" 		       select="/UProtocol/Root/Seq/KSpace/MatrixPE/Value" />
    <xsl:variable name="totoal_pe_sample_rate"   select="/UProtocol/Root/Seq/KSpace/OverSamplingPE/Value div 100 +1" />
    <xsl:variable name="matrix_pe_real" 	       select="/UProtocol/Root/IRIP/FromSeq/KSpace/FTLengthPE/Value" />
    <xsl:variable name="partial_pe_lines" 	     select="/UProtocol/Root/IRIP/FromSeq/KSpace/PartialPELines/Value" />
    <xsl:variable name="rope_interpolation" 	   select="/UProtocol/Root/IRIP/FromUI/Interpolation/Value" />
    
    <!-- EncodingSpace 3rd dimension - SlicePhaseEncoding(SPE) -->
		<xsl:variable name="matrix_spe_ui" 		       select="/UProtocol/Root/Seq/KSpace/MatrixSPE/Value" />
    <xsl:variable name="slice_per_slab" 	       select="/UProtocol/Root/Seq/KSpace/SlicePerSlab/Value" />
    <xsl:variable name="totoal_spe_sample_rate"  select="/UProtocol/Root/Seq/KSpace/OverSamplingSPE/Value div 100 +1" />
    <xsl:variable name="slab_interpolation" 	   select="/UProtocol/Root/Seq/KSpace/SlabInterpolation/Value" />
    <xsl:variable name="partial_spe_lines" 	     select="/UProtocol/Root/IRIP/FromSeq/KSpace/PartialSPELines/Value" />
    <xsl:variable name="matrix_spe_real" 	       select="/UProtocol/Root/IRIP/FromSeq/KSpace/FTLengthSPE/Value" />
 
    <!-- EncodingSpace - FieldOfView(FOV) -->
		<xsl:variable name="fov_ro_ui" 			         select="/UProtocol/Root/Seq/GLI/CommonPara/FOVro/Value" />
		<xsl:variable name="fov_pe_ui" 			         select="/UProtocol/Root/Seq/GLI/CommonPara/FOVpe/Value" />
    <xsl:variable name="thickness"  		         select="/UProtocol/Root/Seq/GLI/CommonPara/Thickness/Value" />

    <!-- EncodingSpace - FastImaging (No=0,Fast1D=1,Fast2D=10,uCS2D=11,Fast2DT=20,4DMRA-33) -->
		<xsl:variable name="is_ppa_on"               select="not(/UProtocol/Root/Seq/PPA/Method/Value=0)" />
    <xsl:variable name="fast_method"             select="/UProtocol/Root/Seq/PPA/Method/Value" />
		<xsl:variable name="ppa_factor_1D" 	         select="/UProtocol/Root/IRIP/FromSeq/PPA/PPAFactorPE/Value" />  
    <xsl:variable name="acc_factor_pe" 	         select="/UProtocol/Root/Seq/PPA/PPAFactorPE/Value" />
	  <xsl:variable name="acc_factor_spe" 	       select="/UProtocol/Root/Seq/PPA/PPAFactorSPE/Value" />
    <xsl:variable name="acc_factor_net" 	       select="/UProtocol/Root/Seq/PPA/CombinedAccel/Value" />

    <!-- EncodingLimits - Slice, Repetition, Phase, Contrast, Average, Segment -->
    <xsl:variable name="MultiSliceMode"          select="/UProtocol/Root/Seq/KSpace/MultiSliceMode/Value" />
    <xsl:variable name="slice"                   select="/UProtocol/Root/Seq/GLI/SliceGroup/ss0/NumberOfSlice/Value"/>
    <xsl:variable name="repetition"              select="/UProtocol/Root/Seq/Basic/Repetition/Value"/>
    <xsl:variable name="contrast"                select="/UProtocol/Root/Seq/Basic/Contrast/Value"/>
    <xsl:variable name="segment"                 select="/UProtocol/Root/Seq/KSpace/Segments/Value"/>
    <xsl:variable name="average"                 select="/UProtocol/Root/Seq/KSpace/Average/Value"/>
    
    <!-- xsl:variable name="phase"               select="" /-->
    
    <!-- variable end -->
    
    <ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xs="http://www.w3.org/2001/XMLSchema" xsi:schemaLocation="http://www.ismrm.org/ISMRMRD ismrmrd.xsd" >
      <measurementInformation>
        <measurementID>
          <xsl:value-of select="/UProtocol/Root/MeasUID/Value"/>
        </measurementID>
        <!-- TODO read from patient parameter and pass as xslt param ? -->
        <patientPosition>
          <xsl:value-of select="$patient_position"/>
        </patientPosition>
        <protocolName>
          <xsl:value-of select="/UProtocol/Header/ProtName"/>
        </protocolName>
      </measurementInformation>
      <acquisitionSystemInformation>
        <systemVendor>UIH</systemVendor>
        <systemModel>N.A</systemModel>
        <systemFieldStrength_T>
          <xsl:value-of select="/UProtocol/Root/SysInfo/TX/NucleusInfo/ss0/Frequency/Value div (42.58 * 1000000)"/>
        </systemFieldStrength_T>
        <receiverChannels>
          <!-- the xslt 2.0 whichi is not support by msxml6 -->
          <!--<xsl:value-of select="sum(/UProtocol/Root/CoilSelection/Para_Array[@ParaTag='SelectedElementGroupInfo']/*/Para_String[@ParaTag='RxChannelID']/Value/(string-length(normalize-space()-string-length(translate(normalize-space(),';','')))" />-->
          <!-- the xslt 1.0 for-each trick whichi is support by msxml6 -->
          <xsl:variable name="all_rxchannelids">
            <xsl:for-each select="UProtocol/Root/CoilSelection/SelectedElementGroupInfo/*/RxChannelID/Value">
              <xsl:value-of select="."/>
            </xsl:for-each>
          </xsl:variable>
          <xsl:value-of select="string-length(normalize-space($all_rxchannelids))-string-length(translate($all_rxchannelids,';',''))"/>
        </receiverChannels>
      </acquisitionSystemInformation>
      <experimentalConditions>
        <H1resonanceFrequency_Hz>
          <xsl:value-of select="/UProtocol/Root/SysInfo/TX/NucleusInfo/ss0/Frequency/Value"/>
        </H1resonanceFrequency_Hz>
      </experimentalConditions>
      <encoding>
        <encodedSpace>
          <matrixSize>
            <x>
              <xsl:value-of select="$matrix_ro_ui * 2"/>
            </x>
            <y>
              <xsl:value-of select="$matrix_pe_real"/>
            </y>
            <z>
              <xsl:choose>
                <xsl:when test="$dimension = $kDIMENSION_3D">
                  <xsl:value-of select="ceiling($slice_per_slab * $totoal_spe_sample_rate)"/>
                </xsl:when>
                <xsl:otherwise>1</xsl:otherwise>
              </xsl:choose>
            </z>
          </matrixSize>
          <fieldOfView_mm>
            <x>
              <xsl:value-of select="$fov_ro_ui * 2"/>
            </x>
            <y>
              <xsl:value-of select="$fov_pe_ui * $totoal_pe_sample_rate"/>
            </y>
            <z>
              <xsl:choose>
                <xsl:when test="$dimension = $kDIMENSION_3D ">
                  <xsl:value-of select=" $thickness * $totoal_spe_sample_rate * $slice_per_slab"/>
                </xsl:when>
                <xsl:otherwise>
                  <xsl:value-of select=" $thickness "/>
                </xsl:otherwise>
              </xsl:choose>
            </z>
          </fieldOfView_mm>
        </encodedSpace>
        <reconSpace>
          <matrixSize>
            <xsl:choose>
              <xsl:when test="$rope_interpolation = $kINT_1x">
                <x>
                  <xsl:value-of select="$matrix_ro_ui"/>
                </x>
              </xsl:when>
              <xsl:otherwise>
                <x>
                  <xsl:value-of select="$matrix_ro_ui * 2"/>
                </x>
              </xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
              <xsl:when test="$rope_interpolation = $kINT_1x">
                <y>
                  <xsl:value-of select="$matrix_pe_ui"/>
                </y>
              </xsl:when>
              <xsl:otherwise>
                <y>
                  <xsl:value-of select="$matrix_pe_ui * 2"/>
                </y>
              </xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
              <xsl:when test="$dimension = $kDIMENSION_3D">
                <z>
                  <xsl:value-of select="$matrix_spe_ui"/>
                </z>
              </xsl:when>
              <xsl:otherwise>
                <z>1</z>
              </xsl:otherwise>
            </xsl:choose>
          </matrixSize>
          <fieldOfView_mm>
            <x>
              <xsl:value-of select="$fov_ro_ui"/>
            </x>
            <y>
              <xsl:value-of select="$fov_pe_ui"/>
            </y>
            <z>
              <xsl:choose>
                <xsl:when test="$dimension=$kDIMENSION_3D">
                  <xsl:value-of select=" $thickness * $slice_per_slab * $totoal_spe_sample_rate "/>
                </xsl:when>
                <xsl:otherwise>
                  <xsl:value-of select=" $thickness"/>
                </xsl:otherwise>
              </xsl:choose>
            </z>
          </fieldOfView_mm>
        </reconSpace>
        
        <encodingLimits>
         
          <kspace_encoding_step_1>
            <minimum>0</minimum>
            <maximum>
              <xsl:value-of select="$matrix_pe_real  - 1"/>
            </maximum>
            <center>
              <xsl:value-of select="floor($matrix_pe_real div 2)"/>
            </center>
          </kspace_encoding_step_1>
 
           <kspace_encoding_step_2>
                <xsl:choose>
                  <xsl:when test="$dimension=$kDIMENSION_3D">
                    <minimum>
                    <xsl:value-of select="floor( ($matrix_spe_real - floor($slice_per_slab * $totoal_spe_sample_rate)) div 2)"/>
                    </minimum>
                  </xsl:when>
                  <xsl:otherwise>
                    <minimum>0</minimum>
                  </xsl:otherwise>
                </xsl:choose>
             
              <xsl:choose>
                <xsl:when test="$dimension=$kDIMENSION_3D">
                  <maximum>
                    <xsl:value-of select="$partial_spe_lines - 1"/>
                  </maximum>
                </xsl:when>
                <xsl:otherwise>
                  <maximum>0</maximum>
                </xsl:otherwise>
              </xsl:choose>

              <xsl:choose>
                <xsl:when test="$dimension=$kDIMENSION_3D">
                  <center>
                    <xsl:value-of select="floor($matrix_spe_real div 2)"/>
                  </center>
                </xsl:when>
                <xsl:otherwise>
                  <center>0</center>
                </xsl:otherwise>
              </xsl:choose>
            </kspace_encoding_step_2>
          
          <!-- Todo: multi slice 2d -->
          <!-- Todo: SMS (Simultaneous MultiSlice) -->
          <slice>
            <minimum>0</minimum>
            <xsl:choose>
              <xsl:when test="$dimension=$kDIMENSION_3D">
                <maximum>0</maximum>
              </xsl:when>
              <xsl:otherwise>
                <maximum>
                  <xsl:value-of select="$slice - 1"/>
                </maximum>
              </xsl:otherwise>
            </xsl:choose>
            <center>0</center>
          </slice>
          
          <repetition>
            <minimum>0</minimum>
            <maximum>0</maximum>
            <center>0</center>
          </repetition>
          
          <segment>
            <minimum>0</minimum>
            <maximum>
              <xsl:value-of select="$segment - 1"/>
            </maximum>
            <center>0</center>
          </segment>
          
          <contrast>
            <minimum>0</minimum>
            <maximum>
              <xsl:value-of select="$contrast - 1"/>
            </maximum>
            <center>0</center>
          </contrast>
          
					<phase>
						<minimum>0</minimum>
						<maximum>
							<xsl:value-of select="$repetition - 1"/>
						</maximum>
						<center>0</center>
					</phase>
          
          <!-- done by uih or gadgetron?-->
          <average>
            <minimum>0</minimum>
            <maximum>
              <xsl:value-of select="$average - 1"/>
            </maximum>
            <center>0</center>
          </average>
        </encodingLimits>
        
        <parallelImaging>
          <accelerationFactor>
            <xsl:choose>
              <xsl:when test="$is_ppa_on">
                <kspace_encoding_step_1>
                  <xsl:choose>
                    <xsl:when test="$fast_method = $kPPAMethod_uCS">
                      <xsl:value-of select="$acc_factor_net"></xsl:value-of>
                    </xsl:when>
                    <xsl:otherwise>
                      <xsl:value-of select="$acc_factor_pe"></xsl:value-of>
                    </xsl:otherwise>
                  </xsl:choose>
                </kspace_encoding_step_1>
              </xsl:when>
              <xsl:otherwise>
                <kspace_encoding_step_1>1</kspace_encoding_step_1>
              </xsl:otherwise>
            </xsl:choose>
            <xsl:choose>
              <xsl:when test="$is_ppa_on">
                <kspace_encoding_step_2>
                  <xsl:choose>
                    <xsl:when test="$fast_method = $kPPAMethod_uCS">
                      <xsl:value-of select="$acc_factor_net"></xsl:value-of>
                    </xsl:when>
                    <xsl:otherwise>
                      <xsl:value-of select="$acc_factor_spe"></xsl:value-of>
                    </xsl:otherwise>
                  </xsl:choose>
                  </kspace_encoding_step_2>
              </xsl:when>
              <xsl:otherwise>
                <kspace_encoding_step_2>1</kspace_encoding_step_2>
              </xsl:otherwise>
            </xsl:choose>
          </accelerationFactor>
          
          <!--
      <xs:enumeration value="embedded"/> 		classic grappa
      <xs:enumeration value="interleaved"/>  	tgrappa
      <xs:enumeration value="separate"/>		acq ref data need before subsample slice kspace data
      <xs:enumeration value="other"/>			acq ref data need before subsample slice kspace data?   
           -->
          <xsl:choose>
            <xsl:when test="$fast_method = $kPPAMethod_Fast2DT">
              <calibrationMode>interleaved</calibrationMode>
            </xsl:when>
            <xsl:otherwise>
              <calibrationMode>embedded</calibrationMode>
            </xsl:otherwise>
          </xsl:choose>
  
          <!-- 2DT dynamic imaging -->
          <interleavingDimension>phase</interleavingDimension>
          
        </parallelImaging>
        <!-- Here fill ttrajectory to cartesian-->
        <trajectory>cartesian</trajectory>
      </encoding>
      
      <sequenceParameters>
        <TR>
          <xsl:value-of select="//TR/Value div 1000.0"/>
        </TR>
        <xsl:choose>
          <xsl:when test="//TE/Value > 0">
            <TE>
              <xsl:value-of select="//TE/Value div 1000.0"/>
            </TE>
          </xsl:when>
          <xsl:otherwise>
            <TE>0</TE>
          </xsl:otherwise>
        </xsl:choose>
        <xsl:choose>
          <xsl:when test="//TI/Value > 0">
            <TI>
              <xsl:value-of select="//TI/Value div 1000.0"/>
            </TI>
          </xsl:when>
          <xsl:otherwise>
            <TI>0</TI>
          </xsl:otherwise>
        </xsl:choose>
      </sequenceParameters>
      
    </ismrmrdHeader>
	</xsl:template>
</xsl:stylesheet>
"""